//! CPU `Race` → GPU lane state. One line per field of `layout_spec.rs`; `LaneState::missing` catches a field
//! left out here.

use super::layout::*;
use crate::data::param::ParamDefine;
use crate::rng::RaceRandom;
use crate::data::CourseLaneAnim;
use crate::sim::ai::lane::LanePlanner;
use crate::sim::ai::order::OrderChangeCounter;
use crate::sim::ai::position_keep::{KeepRule, PositionKeepStrategyKind};
use crate::sim::defs::{RunningStyle, SkillModifierParam};
use crate::sim::skill::modifier::{SkillParamModifier, SkillParamModifierKind, SkillModifierReceiver};
use crate::sim::race::Race;

struct Counter {
    cooldown: Field,
    cool_down_time: Field,
    phase_start: Field,
    phase_middle: Field,
    phase_end_after: Field,
    corner: Field,
    corner_phase_end_after: Field,
    last_spurt: Field,
    distance1: Field,
    final_corner_after: Field,
    changed: Field,
}

const UP: Counter = Counter {
    cooldown: H_OCU_COOLDOWN,
    cool_down_time: H_OCU_COOL_DOWN_TIME,
    phase_start: H_OCU_PHASE_START,
    phase_middle: H_OCU_PHASE_MIDDLE,
    phase_end_after: H_OCU_PHASE_END_AFTER,
    corner: H_OCU_CORNER,
    corner_phase_end_after: H_OCU_CORNER_PHASE_END_AFTER,
    last_spurt: H_OCU_LAST_SPURT,
    distance1: H_OCU_DISTANCE1,
    final_corner_after: H_OCU_FINAL_CORNER_AFTER,
    changed: H_OCU_CHANGED,
};

const DOWN: Counter = Counter {
    cooldown: H_OCD_COOLDOWN,
    cool_down_time: H_OCD_COOL_DOWN_TIME,
    phase_start: H_OCD_PHASE_START,
    phase_middle: H_OCD_PHASE_MIDDLE,
    phase_end_after: H_OCD_PHASE_END_AFTER,
    corner: H_OCD_CORNER,
    corner_phase_end_after: H_OCD_CORNER_PHASE_END_AFTER,
    last_spurt: H_OCD_LAST_SPURT,
    distance1: H_OCD_DISTANCE1,
    final_corner_after: H_OCD_FINAL_CORNER_AFTER,
    changed: H_OCD_CHANGED,
};

fn pack_counter(s: &mut LaneState, h: usize, f: &Counter, c: &OrderChangeCounter) {
    for (k, &cd) in c.count_target_cool_down.iter().enumerate() {
        s.f32(f.cooldown.horse_k(h, k), cd);
    }
    s.f32(f.cool_down_time.horse(h), c.count_cool_down_time);
    s.i32(f.phase_start.horse(h), c.change_count_phase_start);
    s.i32(f.phase_middle.horse(h), c.change_count_phase_middle);
    s.i32(f.phase_end_after.horse(h), c.change_count_phase_end_after);
    s.i32(f.corner.horse(h), c.change_count_corner);
    s.i32(f.corner_phase_end_after.horse(h), c.change_count_corner_phase_end_after);
    s.i32(f.last_spurt.horse(h), c.change_count_last_spurt);
    s.i32(f.distance1.horse(h), c.change_count_distance1);
    s.i32(f.final_corner_after.horse(h), c.change_count_final_corner_after);
    s.bool(f.changed.horse(h), c.is_order_change_per_frame);
}

/// The lane state of `race`.
pub fn pack(race: &Race) -> LaneState {
    let n = race.horses.len();
    let mut s = LaneState::new(n);
    s.i32(R_NUM_HORSES.race(), n as i32);
    s.u32(R_GPU_FAULT.race(), 0);
    let order = &race.field.standings.horse_array_by_order;
    for k in 0..MAX_HORSES as usize {
        let c = order.get(k);
        s.i32(R_ORDER_HORSE.race_k(k), c.map_or(0, |c| c.horse as i32));
        s.f32(R_ORDER_DISTANCE.race_k(k), c.map_or(0.0, |c| c.distance));
    }
    let f = &race.field;
    s.bool(R_FRAME_ALL_FINISHED.race(), false);
    s.i32(R_NUM_RACE_HORSES.race(), race.info.num_race_horses);
    s.i32(R_LAST_ORDER.race(), f.last_order);
    s.bool(R_IS_FINISH_ORDER_CALCULATED.race(), f.is_finish_order_calculated);
    s.bool(R_IS_DEFEAT_CHECKED.race(), f.is_defeat_checked);
    for k in 0..MAX_HORSES as usize {
        s.i32(R_FINISH_ORDER_HORSE.race_k(k), f.horse_index_by_finish_order.get(k).map_or(0, |&h| h as i32));
    }
    let cs = &race.info.race_course_set;
    let (min, max) = (cs.finish_time_min as f32 * 0.0001, cs.finish_time_max as f32 * 0.0001);
    let (rmin, rmax) = (cs.finish_time_min_random_range as f32 * 0.0001, cs.finish_time_max_random_range as f32 * 0.0001);
    s.f32(R_FINISH_TIME_MIN.race(), min);
    s.f32(R_FINISH_TIME_MAX.race(), max);
    s.f32(R_FINISH_TIME_CLAMP_LOW.race(), RaceRandom::new(race.info.random_seed).random_float_range(-rmin, rmin) + min);
    s.f32(R_FINISH_TIME_CLAMP_HIGH.race(), RaceRandom::new(race.info.random_seed).random_float_range(-rmax, rmax) + max);
    let list = |s: &mut LaneState, len: Field, items: Field, style: usize, v: &[usize]| {
        s.i32(len.race_k(style), v.len() as i32);
        for k in 0..MAX_HORSES as usize {
            s.i32(items.race_k(style * MAX_HORSES as usize + k), v.get(k).map_or(0, |&h| h as i32));
        }
    };
    for st in 0..5 {
        s.i32(R_STYLE_COUNT.race_k(st), f.same_running_style_count[st]);
        list(&mut s, R_STYLE_LEN, R_STYLE_HORSE, st, &f.same_running_style_horses[st]);
        list(&mut s, R_STYLE_RUSHED_LEN, R_STYLE_RUSHED_HORSE, st, &f.rushed_horses_by_style[st]);
        s.i32(R_STYLE_TOP_HORSE.race_k(st), f.running_style_top_order_horses[st].map_or(-1, |h| h as i32));
        s.i32(R_STYLE_LAST_HORSE.race_k(st), f.running_style_last_order_horses[st].map_or(-1, |h| h as i32));
    }
    let sv = &f.scan_view;
    for k in 0..MAX_HORSES as usize {
        s.f32(R_SV_DISTANCE.race_k(k), sv.distance.0[k]);
        s.f32(R_SV_LANE.race_k(k), sv.lane_distance.0[k]);
        s.f32(R_SV_LAST_SPEED.race_k(k), sv.last_speed.0[k]);
    }
    s.u32(R_SV_STRAIGHT_LAST.race(), sv.is_straight_last);
    s.f32(R_COURSE_SECTION_DISTANCE.race(), race.info.course_section_distance);
    for k in 0..2 {
        s.i32(R_COMPETE_TOP_START_COUNT.race_k(k), f.compete_top.start_count[k]);
    }
    s.i32(R_MOST_FORWARD_RUNNING_STYLE.race(), f.most_forward_running_style as i32);
    let opt = |o: Option<usize>| o.map_or(-1, |j| j as i32);
    s.i32(R_PM_HORSE.race(), opt(f.pacemaker.horse));
    s.i32(R_PM_SECTION.race(), f.pacemaker.section);
    s.i32(R_PM_SECTION_PICK.race(), opt(f.pacemaker.section_pick));
    s.i32(R_PM_NOT_MOST_FORWARD_COUNT.race(), f.pacemaker.not_most_forward_count);
    s.bool(R_PM_OVERRIDING.race(), f.pacemaker.overriding);
    s.i32(R_RUSHED_LEN.race(), f.rushed_horses.len() as i32);
    for k in 0..MAX_HORSES as usize {
        s.i32(R_RUSHED_HORSE.race_k(k), f.rushed_horses.get(k).map_or(0, |&h| h as i32));
    }
    for (k, &w) in race.rng.random.words().iter().enumerate() {
        s.u32(R_RNG.race_k(k), w);
    }
    s.f32(R_ACCUMULATE_TIME.race(), race.accumulate_time_since_start);
    s.f32(R_WORLD_PER_METRE.race(), race.world_per_metre);
    s.f32(R_BASE_SPEED.race(), race.info.base_speed);
    let pd = &race.info.phase_distances;
    s.f32(R_PHASE_MIDDLE_START.race(), pd.phase_middle_start_distance);
    s.f32(R_PHASE_END_START.race(), pd.phase_end_start_distance);
    s.f32(R_PHASE_LAST_START.race(), pd.phase_last_start_distance);
    s.f32(R_PHASE_COURSE_DISTANCE.race(), pd.course_distance);
    s.i32(R_COURSE_DISTANCE.race(), race.info.race_course_set.distance);
    s.i32(R_COURSE_DISTANCE_TYPE.race(), race.info.course_distance_type);
    s.f32(R_INFO_LANE_DISTANCE_MAX.race(), race.info.lane_distance_max);
    s.f32(R_FIRST_MOVE_LANE_POINT.race(), race.course.event_param.first_move_lane_point_distance);
    s.bool(R_PRE_2026_07.race(), race.data.era.pre_2026_07_courses);
    s.u32(R_COURSE_BASE.race(), 0);
    s.u32(R_SKILL_BASE.race(), 0);
    s.u32(R_NEXT_MODIFIER_ID.race_k(0), race.next_modifier_id as u32);
    s.u32(R_NEXT_MODIFIER_ID.race_k(1), (race.next_modifier_id >> 32) as u32);
    s.f32(R_DISTANCE_DIFF_MAX.race(), race.distance_diff_max);
    s.bool(R_IS_STORY.race(), race.info.race_type == crate::sim::defs::race_type::STORY);
    s.bool(R_RACE_DONE.race(), false);
    for (h, x) in race.horses.iter().enumerate() {
        s.f32(H_DISTANCE.horse(h), x.distance);
        s.i32(H_CUR_ORDER.horse(h), x.cur_order);
        s.i32(H_PREV_ORDER.horse(h), x.prev_order);
        s.bool(H_IS_FINISHED.horse(h), x.is_finished);
        s.i32(H_PHASE.horse(h), x.phase as i32);
        s.i32(H_CUR_CORNER.horse(h), x.cur_corner);
        s.bool(H_IS_FINAL_CORNER.horse(h), x.is_final_corner);
        s.f32(H_LS_START_DISTANCE.horse(h), x.ai.last_spurt_start_distance);
        s.i32(H_LS_CALC_RESULT.horse(h), x.ai.last_spurt_calc_result);
        s.f32(H_NO_ORDER_UP_TIME.horse(h), x.overtake_target_have_no_order_up_continue_time);
        s.f32(H_NO_ORDER_DOWN_TIME.horse(h), x.overtake_target_have_no_order_down_continue_time);
        pack_counter(&mut s, h, &UP, &x.order_change_counter_up);
        pack_counter(&mut s, h, &DOWN, &x.order_change_counter_down);
        s.f32(H_LANE_DISTANCE.horse(h), x.lane_distance);
        s.f32(H_LANE_DISTANCE_PER_IN_ALL_HORSES.horse(h), x.lane_distance_per_in_all_horses);
        s.f32(H_LAST_ORDER_TIME.horse(h), x.last_order_time);
        s.bool(H_IS_TEMPTATION.horse(h), x.ai.temptation.is_temptation);
        s.f32(H_IN_FRONT_NEAR_TIME.horse(h), x.in_front_horse_near_continue_time);
        s.f32(H_IN_FRONT_NEAR_LANE_TIME.horse(h), x.in_front_horse_near_lane_continue_time);
        s.f32(H_BEHIND_NEAR_TIME.horse(h), x.behind_horse_near_continue_time);
        s.f32(H_BEHIND_NEAR_LANE_TIME.horse(h), x.behind_horse_near_lane_continue_time);
        assert_eq!(x.behind_near_lane_times.len(), race.data.param.Skill.BehindNearParamArray.len());
        for k in 0..MAX_BEHIND_NEAR_PARAMS as usize {
            s.f32(H_BEHIND_NEAR_LANE_TIMES.horse_k(h, k), x.behind_near_lane_times.get(k).copied().unwrap_or(0.0));
        }
        s.f32(H_FINISH_TIME_SCALED.horse(h), x.finish_time_scaled);
        s.f64(H_FINISH_TIME_FOR_ORDER.horse(h), x.finish_time_scaled_for_finish_order_calc);
        s.i32(H_FINISH_ORDER.horse(h), x.finish_order);
        s.f32(H_FINISH_TIME_DIFF_FROM_PREV.horse(h), x.finish_time_diff_from_prev_horse);
        s.i32(H_DEFEAT.horse(h), x.defeat);
        s.i32(H_RUNNING_STYLE.horse(h), x.horse_data.running_style as i32);
        s.i32(H_MOTIVATION.horse(h), x.horse_data.motivation);
        s.i32(H_ACTIVE_PROPER_DISTANCE.horse(h), x.horse_data.active_proper_distance);
        s.i32(H_ACTIVE_PROPER_GROUND_TYPE.horse(h), x.horse_data.active_proper_ground_type);
        s.bool(H_IS_HP_EMPTY_ON_RACE.horse(h), x.is_hp_empty_on_race);
        s.i32(H_TEMPTATION_COUNT.horse(h), x.ai.temptation.temptation_count);
        s.i32(H_GUTS_ORDER.horse(h), x.guts_order);
        s.i32(H_SKILL_COUNT.horse(h), x.skill_manager.skills.len() as i32);
        s.f32(H_BLOCK_FRONT_ACCUMULATE_TIME.horse(h), x.block_front_accumulate_time);
        s.f32(H_LAST_SPEED.horse(h), x.last_speed);
        s.bool(H_IS_STRAIGHT.horse(h), x.is_straight);
        s.f32(H_LANE_DISTANCE_MAX.horse(h), x.lane_distance_max);
        s.f32(H_CORNER_END_DISTANCE.horse(h), x.corner_end_distance);
        s.f32(H_MOVE_DISTANCE.horse(h), x.move_distance);
        s.f32(H_BASE_VISIBLE_DISTANCE.horse(h), x.base_visible_distance);
        s.f32(H_AFTER_GOAL_TIME.horse(h), x.ai.after_goal_time);
        s.i32(H_ACTIVE_LANE_PLANNER.horse(h), x.ai.active_lane_planner as i32);
        let t = &*x.ai.around;
        // Columns past the last horse are never read; the CPU's AVX2 scan writes them eight at a time, the GPU not.
        let col = |c: &[f32], k: usize| if k < n { c[k] } else { 0.0 };
        for k in 0..MAX_HORSES as usize {
            s.f32(H_AR_DISTANCE_GAP.horse_k(h, k), col(&t.distance_gap.0, k));
            s.f32(H_AR_DISTANCE_GAP_ABS.horse_k(h, k), col(&t.distance_gap_abs.0, k));
            s.f32(H_AR_LANE_GAP.horse_k(h, k), col(&t.lane_gap.0, k));
            s.f32(H_AR_LANE_GAP_ABS.horse_k(h, k), col(&t.lane_gap_abs.0, k));
            s.f32(H_AR_SPEED_GAP.horse_k(h, k), col(&t.speed_gap.0, k));
            s.f32(H_AR_SPEED_GAP_ABS.horse_k(h, k), col(&t.speed_gap_abs.0, k));
            s.f32(H_AR_CF_TIME.horse_k(h, k), col(&t.compete_fight_candidate_continue_time.0, k));
        }
        s.u32(H_AR_BACK.horse(h), t.back);
        s.u32(H_AR_OTHERS.horse(h), t.others);
        let ints = |s: &mut LaneState, f: Field, v: &[usize]| {
            assert!(v.len() <= MAX_HORSES as usize);
            for k in 0..MAX_HORSES as usize {
                s.i32(f.horse_k(h, k), v.get(k).map_or(0, |&j| j as i32));
            }
        };
        ints(&mut s, H_NEAR_HORSES, &x.ai.near_horses);
        s.i32(H_NEAR_HORSE_COUNT.horse(h), x.ai.near_horse_count);
        ints(&mut s, H_VISIBLE_HORSES, &x.ai.visibility.visible_horses);
        s.i32(H_VISIBLE_HORSE_COUNT.horse(h), x.ai.visibility.visible_horse_count);
        s.i32(H_CF_NEAR_LEN.horse(h), x.compete_fight_near.len() as i32);
        ints(&mut s, H_CF_NEAR, &x.compete_fight_near);
        let opt = |o: Option<usize>| o.map_or(-1, |j| j as i32);
        s.i32(H_BLOCK_HORSE.horse(h), opt(x.ai.block_horse));
        s.i32(H_BLOCK_HORSE_IN.horse(h), opt(x.ai.block_horse_in));
        s.i32(H_BLOCK_HORSE_OUT.horse(h), opt(x.ai.block_horse_out));
        s.f32(H_IN_LANE_SPACE.horse(h), x.ai.in_lane_space);
        s.f32(H_OUT_LANE_SPACE.horse(h), x.ai.out_lane_space);
        s.bool(H_IS_AROUND_HORSE_INITIALIZED.horse(h), x.ai.is_around_horse_initialized);
        s.bool(H_IS_SURROUND_OUT.horse(h), x.ai.is_surround_out);
        s.bool(H_IS_SURROUND_AHEAD.horse(h), x.ai.is_surround_ahead);
        s.bool(H_IS_SURROUND_BEHIND.horse(h), x.ai.is_surround_behind);
        pack_modifiers(&mut s, h, &x.skill_modifier_receiver);
        s.i32(H_RUNNING_STYLE_EX.horse(h), x.horse_data.running_style_ex as i32);
        s.f32(H_HP.horse(h), x.hp);
        s.f32(H_MAX_HP.horse(h), x.max_hp);
        s.f32(H_BASE_GUTS_ADJUSTED.horse(h), x.base_guts_adjusted);
        s.f32(H_PROPER_RUNNING_STYLE_COEF.horse(h), x.proper_running_style_coef);
        s.bool(H_IS_COMPETE_TOP.horse(h), x.is_compete_top);
        s.i32(H_COMPETE_TOP_COUNT.horse(h), x.compete_top_count);
        s.i32(H_COMPETE_TOP_GROUP.horse(h), x.compete_top_group);
        s.f32(H_COMPETE_TOP_REMAIN_TIME.horse(h), x.compete_top_remain_time);
        s.bool(H_IS_COMPETE_FIGHT.horse(h), x.is_compete_fight);
        s.i32(H_COMPETE_FIGHT_COUNT.horse(h), x.compete_fight_count);
        s.i32(H_COMPETE_FIGHT_GROUP.horse(h), x.compete_fight_group);
        s.f32(H_CONSERVED_POWER.horse(h), x.ai.conserve_power.conserved_power);
        s.f32(H_CONSERVE_ACTIVITY_COEF.horse(h), x.ai.conserve_power.activity_coef);
        let lp = match &x.ai.lane_planner {
            LanePlanner::Race(c) => c.is_force_check_next_lane_one_time,
            LanePlanner::Fixed(c) => c.is_force_check_next_lane_one_time,
        };
        s.bool(H_LP_FORCE_CHECK.horse(h), lp);
        s.bool(H_AGL_FORCE_CHECK.horse(h), x.ai.after_goal_lane.is_force_check_next_lane_one_time);
        pack_update_horses(&mut s, race, h);
    }
    s
}

/// The fields UpdateHorses adds.
fn pack_update_horses(s: &mut LaneState, race: &Race, h: usize) {
    let x = &race.horses[h];
    s.f32(H_RACE_DEFAULT_MIN_SPEED.horse(h), x.race_default_min_speed);
    s.f32(H_BASE_SPEED_ADJUSTED.horse(h), x.base_speed_adjusted);
    s.f32(H_BASE_POW_ADJUSTED.horse(h), x.base_pow_adjusted);
    s.f32(H_BASE_WIZ_ADJUSTED.horse(h), x.base_wiz_adjusted);
    s.f32(H_BASE_WIZ_ADJUSTED_ALT.horse(h), x.base_wiz_adjusted_alt);
    s.u32(H_WIZ_ALT_MECHANICS.horse(h), x.horse_data.race_param.wiz_alt_mechanics as u32);
    s.f32(H_MULTI_HP_SUB.horse(h), x.ground_modifier_param.multi_hp_sub);
    s.f32(H_PROPER_DISTANCE_COEF_SPEED.horse(h), x.proper_distance_coef_speed);
    s.f32(H_PROPER_DISTANCE_COEF_POW.horse(h), x.proper_distance_coef_pow);
    s.f32(H_PROPER_GROUND_TYPE_COEF.horse(h), x.proper_ground_type_coef);
    let c = &x.accel.phase_accel_coef;
    for (k, v) in [c.start, c.middle, c.end, c.last].into_iter().enumerate() {
        s.f32(H_PHASE_ACCEL_COEF.horse_k(h, k), v);
    }
    s.bool(H_IS_EMPTY_IN_MOVE.horse(h), x.is_empty_in_move);
    s.bool(H_IS_START_DASH.horse(h), x.is_start_dash);
    s.f32(H_MIN_SPEED.horse(h), x.min_speed);
    s.f32(H_DELAY_TIME.horse(h), x.delay_time);
    s.f32(H_LAST_SELF_SPEED.horse(h), x.last_self_speed);
    s.f32(H_LANE_MOVE_SPEED.horse(h), x.lane_move_speed);
    s.f32(H_LAST_LANE_MOVE_SPEED.horse(h), x.last_lane_move_speed);
    s.f32(H_MOVE_LANE_CONTINUE_TIME.horse(h), x.move_lane_continue_time);
    s.i32(H_MOVE_LANE_DIR.horse(h), x.move_lane_continue_direction as i32);
    s.f32(H_PREV_LANE_DISTANCE.horse(h), x.prev_lane_distance);
    s.f32(H_LANE_DISTANCE_MIN.horse(h), x.lane_distance_min);
    s.f32(H_PREV_DISTANCE.horse(h), x.prev_distance);
    s.f32(H_PREV_ELAPSED_TIME.horse(h), x.prev_elapsed_time);
    for (k, (v, pv)) in [(x.position.x, x.prev_position.x), (x.position.y, x.prev_position.y), (x.position.z, x.prev_position.z)]
        .into_iter()
        .enumerate()
    {
        s.f32(H_POSITION.horse_k(h, k), v);
        s.f32(H_PREV_POSITION.horse_k(h, k), pv);
    }
    s.i32(H_SLOPE_TYPE.horse(h), x.slope_type as i32);
    s.f32(H_SLOPE_PER.horse(h), x.slope_per);
    s.f32(H_SLOPE_END_DISTANCE.horse(h), x.slope_end_distance);
    s.i32(H_LAST_SPURT_ORDER.horse(h), x.last_spurt_order);
    s.i32(H_FINAL_CORNER_END_ORDER.horse(h), x.final_corner_end_order);
    s.bool(H_IS_LAST_SPURT_PREV.horse(h), x.is_last_spurt_prev);
    s.f32(H_BLOCK_FRONT_CONTINUE_TIME.horse(h), x.block_front_continue_time);
    s.f32(H_BLOCK_SIDE_CONTINUE_TIME.horse(h), x.block_side_continue_time);
    s.f32(H_BLOCK_FRONT_HP_EMPTY_TIME.horse(h), x.block_front_hp_empty_continue_time);
    for k in 0..4 {
        s.f32(H_BLOCK_SIDE_MAX_BY_PHASE.horse_k(h, k), x.block_side_max_continue_time_by_phase[k]);
    }
    s.f32(H_NEAR_AROUND_TIME.horse(h), x.near_horse_around_accumulate_time);
    s.f32(H_OVERTAKE_TARGET_TIME.horse(h), x.overtake_target_continue_time);
    s.i32(H_POSITION_KEEP_COUNT.horse(h), x.position_keep_count);
    for k in 0..ORDER_IN_OUT_TYPES as usize {
        s.bool(H_ORDER_IN_CONTINUE.horse_k(h, k), x.order_in_continue[k]);
        s.bool(H_ORDER_OUT_CONTINUE.horse_k(h, k), x.order_out_continue[k]);
    }
    s.f32(H_FINISH_TIME_RAW.horse(h), x.finish_time_raw);
    let ai = &x.ai;
    s.f32(H_TARGET_SPEED.horse(h), ai.target_speed);
    s.f32(H_TARGET_LANE.horse(h), ai.target_lane);
    s.bool(H_DOWN_SLOPE_MODE.horse(h), ai.is_down_slope_accel_mode);
    s.f32(H_DOWN_SLOPE_CHECK_SEC.horse(h), ai.down_slope_accel_check_sec);
    s.bool(H_REQ_CALC_LS.horse(h), ai.is_req_calc_last_spurt);
    s.bool(H_LS_BY_DISTANCE_REQUESTED.horse(h), ai.is_last_spurt_by_distance_requested);
    s.bool(H_FORCE_IN_MOVE.horse(h), ai.is_force_in_move_enable);
    s.i32(H_LS_CALC_COUNT.horse(h), ai.last_spurt_calc_count);
    s.f32(H_LS_TARGET_SPEED.horse(h), ai.last_spurt_target_speed);
    s.f32(H_FORCE_IN_MOVE_ADD.horse(h), ai.target_speed_calc.force_in_move_add_target_speed);
    s.f32(H_CALC_TARGET_SPEED.horse(h), ai.target_speed_calc.target_speed);
    let ot = &ai.overtake_targets.targets;
    assert!(ot.len() <= MAX_HORSES as usize);
    s.i32(H_OT_COUNT.horse(h), ot.len() as i32);
    for k in 0..MAX_HORSES as usize {
        s.i32(H_OT_TARGETS.horse_k(h, k), ot.get(k).map_or(0, |&j| j as i32));
    }
    s.f32(H_OT_COOL_DOWN.horse(h), ai.overtake_cool_down_time);
    let b = &ai.base_target_speeds;
    assert_eq!(b.sections.len(), BTS_SECTIONS as usize);
    assert_eq!(b.section_change_distance.to_bits(), race.info.course_section_distance.to_bits());
    assert_eq!(b.race_base_speed.to_bits(), race.info.base_speed.to_bits());
    s.bool(H_BTS_INITIALIZED.horse(h), b.is_initialized);
    for (k, e) in b.sections.iter().enumerate() {
        s.f32(H_BTS_SPEED.horse_k(h, k), e.base_target_speed);
        s.f32(H_BTS_RANDOM.horse_k(h, k), e.base_target_speed_random);
        s.i32(H_BTS_PHASE.horse_k(h, k), e.phase as i32);
    }
    let pk = &ai.position_keep;
    assert_eq!(pk.course_distance, race.info.race_course_set.distance);
    assert_eq!(pk.strategies.len(), PK_STRATEGIES as usize);
    s.f32(H_PK_CHECK_START.horse(h), pk.check_start_distance);
    s.f32(H_PK_CHECK_END.horse(h), pk.check_end_distance);
    let (style, force) = match pk.keep_rule {
        KeepRule::Nige { is_force_speed_up } => (RunningStyle::Nige, is_force_speed_up),
        KeepRule::PaceAdjust { course_distance, running_style } => {
            assert_eq!(course_distance, pk.course_distance);
            assert_ne!(running_style, RunningStyle::Nige);
            (running_style, false)
        }
    };
    s.i32(H_PK_STYLE.horse(h), style as i32);
    s.bool(H_PK_FORCE_SPEED_UP.horse(h), force);
    s.bool(H_PK_TEMPTATION_PREV.horse(h), pk.is_temptation_prev);
    s.i32(H_PK_CUR.horse(h), pk.cur_strategy as i32);
    s.f32(H_PK_CONTINUE_DISTANCE.horse(h), pk.strategies[1].continue_distance);
    for (k, st) in pk.strategies.iter().enumerate() {
        assert_eq!(st.mode() as usize, k);
        if k > 0 {
            assert_eq!(st.continue_distance.to_bits(), pk.strategies[1].continue_distance.to_bits());
        }
        s.f32(H_PK_END_DISTANCE.horse_k(h, k), st.end_distance);
        s.f32(H_PK_MULT.horse_k(h, k), st.base_target_speed_multiply);
        if let PositionKeepStrategyKind::PaceUp(ps) | PositionKeepStrategyKind::PaceDown(ps) | PositionKeepStrategyKind::PaceUpEx(ps) =
            st.kind
        {
            assert_eq!((ps.running_style, ps.course_distance), (style, pk.course_distance));
            s.f32(H_PK_PS_MIN.horse_k(h, k - 3), ps.distance_diff_min);
            s.f32(H_PK_PS_MAX.horse_k(h, k - 3), ps.distance_diff_max);
            s.f32(H_PK_PS_TARGET.horse_k(h, k - 3), ps.pace_adjust_target_distance_diff);
        }
    }
    s.f32(H_PK_COOL_DOWN.horse(h), pk.cool_down_sec);
    s.f32(H_PK_CHECK_INTERVAL.horse(h), pk.check_start_interval_sec);
    let t = &ai.temptation;
    s.i32(H_TM_START_SECTION.horse(h), t.temptation_start_section);
    s.f32(H_TM_END_CHECK_SEC.horse(h), t.temptation_end_check_sec);
    s.f32(H_TM_FORCE_END_SEC.horse(h), t.temptation_force_end_sec);
    s.bool(H_TM_SKILL_END_ENABLE.horse(h), t.is_force_end_time_by_skill_enable);
    s.f32(H_TM_SKILL_END_TIME.horse(h), t.force_end_time_by_skill);
    s.i32(H_TM_MODE.horse(h), t.mode as i32);
    s.bool(H_TM_START_ENABLE.horse(h), t.is_temptation_start_enable);
    match &ai.lane_planner {
        LanePlanner::Race(c) => {
            assert_eq!(c.first_move_lane_point_distance.to_bits(), race.course.event_param.first_move_lane_point_distance.to_bits());
            s.bool(H_LP_IS_RACE.horse(h), true);
            s.f32(H_LP_LAST_MOVE_OUT_START.horse(h), c.last_move_out_start_distance);
            s.bool(H_LP_EXTRA_CHECKED.horse(h), c.is_use_extra_move_checked);
            s.bool(H_LP_USE_EXTRA.horse(h), c.is_use_extra_move);
            s.f32(H_LP_EXTRA_LANE.horse(h), c.extra_move_lane);
            s.f32(H_LP_TARGET_LANE.horse(h), c.target_lane);
        }
        LanePlanner::Fixed(c) => {
            assert!(!c.is_use_extra_move);
            s.bool(H_LP_IS_RACE.horse(h), false);
            s.f32(H_LP_LAST_MOVE_OUT_START.horse(h), 0.0);
            s.bool(H_LP_EXTRA_CHECKED.horse(h), false);
            s.bool(H_LP_USE_EXTRA.horse(h), false);
            s.f32(H_LP_EXTRA_LANE.horse(h), 0.0);
            s.f32(H_LP_TARGET_LANE.horse(h), 0.0);
        }
    }
    let cp = &ai.conserve_power;
    s.bool(H_CP_ACTIVE.horse(h), cp.is_active);
    s.f32(H_CP_POWER_STAT.horse(h), cp.power_stat);
    s.f32(H_CP_COOL_DOWN.horse(h), cp.cool_down);
    s.f32(H_CP_BANK_TIMER.horse(h), cp.bank_timer);
    s.bool(H_CP_TEMPTATION_PREV.horse(h), cp.is_temptation_prev);
    s.bool(H_CP_RELEASED.horse(h), cp.is_released);
    s.f32(H_CP_ACCEL.horse(h), cp.accel_value);
    s.f32(H_CP_REMAIN_TIME.horse(h), cp.remain_time);
    s.bool(H_IS_GOOD_START.horse(h), x.is_good_start);
    s.bool(H_IS_BAD_START.horse(h), x.is_bad_start);
    s.i32(H_CUR_STRAIGHT_FRONT_TYPE.horse(h), x.cur_straight_front_type);
    s.bool(H_IS_PARAM_INITIALIZED.horse(h), x.is_param_initialized);
    s.bool(H_IS_SKILL_INITIALIZED.horse(h), x.is_skill_initialized);
    s.bool(H_IS_COMPLETE_FIRST_UPDATE.horse(h), x.is_complete_first_update);
    for k in 0..4 {
        s.i32(H_ACT_COUNT_BY_PHASE.horse_k(h, k), x.activate_skill_count_by_phase[k]);
    }
    for k in 0..2 {
        s.i32(H_ACT_COUNT_BY_DISTANCE.horse_k(h, k), x.activate_skill_count_by_distance[k]);
    }
    s.i32(H_ACT_HEAL_COUNT.horse(h), x.activate_heal_skill_count);
    s.i32(H_ACT_TAG_GROUP_COUNT.horse(h), x.activate_specific_tag_group_skill_count);
    s.i32(H_LAST_EVENT.horse(h), race.event_manager.last_event_by_horse[h]);
    s.i32(H_GROUND_TYPE.horse(h), x.ground_type);
    s.i32(H_CUR_STRAIGHT_TYPE_NUMBER.horse(h), x.cur_straight_type_number);
    s.i32(H_LAST_STRAIGHT_ORDER.horse(h), x.last_straight_order);
}

/// The course buffer block of a race (`race_course_base` points at it): a header, then the lane animation and the
/// overrun animation. All offsets are words from the start of the block.
///
/// Header: main animation offset, overrun animation offset (0 = none), goal_pos xyz, goal_dir xyz, goal_rot xyzw,
/// rotation_fix_rot xyzw, then offset and count of the course events, the straights, the corners and the slopes.
/// Animation: Distance, len X, len Y, len Z, len rotation, then X, Y, Z and rotation xyzw. Event: distance, param
/// type, value count, value offset (the values follow the events). Straight: front type, start, end, front type
/// number. Corner: corner number, start, end, is final. Slope: slope type, start, end, 0.
pub fn course_words(race: &Race) -> Vec<u32> {
    let c = &race.course_lane;
    let mut w = vec![0u32; COURSE_HEADER_WORDS];
    let v = |w: &mut Vec<u32>, at: usize, xs: &[f32]| {
        for (k, &x) in xs.iter().enumerate() {
            w[at + k] = x.to_bits();
        }
    };
    v(&mut w, 2, &[c.goal_pos.x, c.goal_pos.y, c.goal_pos.z]);
    v(&mut w, 5, &[c.goal_dir.x, c.goal_dir.y, c.goal_dir.z]);
    v(&mut w, 8, &[c.goal_rot.x, c.goal_rot.y, c.goal_rot.z, c.goal_rot.w]);
    let r = c.rotation_fix_rot;
    v(&mut w, 12, &[r.x, r.y, r.z, r.w]);
    let anim = |w: &mut Vec<u32>, a: &CourseLaneAnim| -> u32 {
        let at = w.len() as u32;
        let k = &a.key;
        w.extend([a.Distance as u32, k.valueX.len() as u32, k.valueY.len() as u32, k.valueZ.len() as u32, k.rotation.len() as u32]);
        w.extend(k.valueX.iter().chain(&k.valueY).chain(&k.valueZ).map(|x| x.to_bits()));
        w.extend(k.rotation.iter().flat_map(|q| [q.x, q.y, q.z, q.w]).map(|x| x.to_bits()));
        at
    };
    w[0] = anim(&mut w, &c.course_lane_anim);
    if let Some(o) = &c.course_over_run_lane_anim {
        w[1] = anim(&mut w, o);
    }
    let cs = &race.course;
    let events = &cs.event_param.course_event_list;
    let at = w.len();
    w[16] = at as u32;
    w[17] = events.len() as u32;
    w.resize(at + 4 * events.len(), 0);
    for (k, ev) in events.iter().enumerate() {
        let p = &ev.param;
        let vals = w.len() as u32;
        w.extend(p._values.iter().map(|&v| v as u32));
        w[at + 4 * k..at + 4 * k + 4].copy_from_slice(&[p._distance.to_bits(), p._paramType as u32, p._values.len() as u32, vals]);
    }
    w[18] = w.len() as u32;
    w[19] = cs.straights.len() as u32;
    for s in &cs.straights {
        w.extend([s.front_type as u32, s.start_distance.to_bits(), s.end_distance.to_bits(), s.front_type_number as u32]);
    }
    w[20] = w.len() as u32;
    w[21] = cs.corners.len() as u32;
    for x in &cs.corners {
        w.extend([x.corner_number as u32, x.start_distance.to_bits(), x.end_distance.to_bits(), x.is_final_corner as u32]);
    }
    w[22] = w.len() as u32;
    w[23] = cs.slopes.len() as u32;
    for x in &cs.slopes {
        w.extend([x.slope_type as i32 as u32, x.start_distance.to_bits(), x.end_distance.to_bits(), 0]);
    }
    w
}

/// Words before the first animation in a course block.
pub const COURSE_HEADER_WORDS: usize = 24;

/// The params buffer of a data era.
pub fn pack_params(p: &ParamDefine) -> ParamWords {
    let mut w = ParamWords::default();
    let sk = &p.Skill;
    w.f32(P_INFRONT_NEAR_DISTANCE.param(), sk.InfrontHorseNearDistance);
    w.f32(P_INFRONT_NEAR_LANE_DISTANCE.param(), sk.InfrontHorseNearLaneDistance);
    w.f32(P_BEHIND_NEAR_DISTANCE.param(), sk.BehindHorseNearDistance);
    w.f32(P_BEHIND_NEAR_LANE_DISTANCE.param(), sk.BehindHorseNearLaneDistance);
    assert!(sk.BehindNearParamArray.len() <= MAX_BEHIND_NEAR_PARAMS as usize, "raise MAX_BEHIND_NEAR_PARAMS");
    w.i32(P_BEHIND_NEAR_PARAM_COUNT.param(), sk.BehindNearParamArray.len() as i32);
    for k in 0..MAX_BEHIND_NEAR_PARAMS as usize {
        let e = sk.BehindNearParamArray.get(k);
        w.f32(P_BEHIND_NEAR_PARAM_DISTANCE.param_k(k), e.map_or(0.0, |e| e.Distance));
        w.f32(P_BEHIND_NEAR_PARAM_LANE_DISTANCE.param_k(k), e.map_or(0.0, |e| e.LaneDistance));
    }
    let d = &p.Defeat;
    w.i32(P_DEFEAT_SAME_RUNNING_STYLE_PER.param(), d.DefeatSameRunningStylePer);
    w.i32(P_DEFEAT_GUTS_ORDER_PER.param(), d.DefeatGutsOrderPer);
    w.i32(P_DEFEAT_TEMPTATION_COUNT.param(), d.DefeatTemptaionCount);
    w.f32(P_DEFEAT_FRONT_BLOCK_ACCUMULATE_TIME.param(), d.DefeatFrontBlockAccumulateTime);
    w.i32(P_DEFEAT_LOSE_FINISH_ORDER.param(), d.DefeatLoseFinishOrder);
    w.i32(P_DEFEAT_LOSE_BASHIN_DIFF.param(), d.DefeatLoseBashinDiff);
    w.i32(P_DEFEAT_PROPER_DISTANCE.param(), d.DefeatProperDistance);
    w.i32(P_DEFEAT_PROPER_GROUND.param(), d.DefeatProperGround);
    w.i32(P_DEFEAT_MOTIVATION.param(), d.DefeatMotivation);
    w.f32(P_SURROUND_SIDE_DISTANCE.param(), p.Surrounded.DistanceGapAbs1);
    w.f32(P_SURROUND_SIDE_LANE.param(), p.Surrounded.LaneGap1);
    w.f32(P_SURROUND_DISTANCE.param(), p.Surrounded.DistanceGap2);
    w.f32(P_SURROUND_LANE.param(), p.Surrounded.LaneGapAbs2);
    w.f32(P_NEAR_DISTANCE.param(), p.Near.DistanceThreshold);
    w.f32(P_NEAR_LANE.param(), p.Near.LaneDistanceThreshold);
    w.f32(P_COMPETE_FIGHT_LANE_GAP.param(), p.CompeteFight.LaneGap);
    w.f32(P_COMPETE_FIGHT_DISTANCE_GAP.param(), p.CompeteFight.DistanceGap);
    w.f32(P_SIDE_BLOCK_DISTANCE_GAP.param(), p.Block.SideBlockDistanceGap);
    w.f32(P_SIDE_BLOCK_LANE_GAP.param(), p.Block.SideBlockLaneGap);
    w.f32(P_FRONT_BLOCK_DISTANCE_GAP.param(), p.Block.FrontBlockDistanceGap);
    w.f32(P_FRONT_BLOCK_LANE_GAP.param(), p.Block.FrontBlockLaneGap);
    let ct = &p.CompeteTop;
    w.f32(P_CT_CHECK_START_DISTANCE.param(), ct.CheckStartDistance);
    w.i32(P_CT_CHECK_END_SECTION.param(), ct.CheckEndSection);
    w.i32(P_CT_END_SECTION.param(), ct.EndSection);
    w.i32(P_CT_NIGE_COUNT.param(), ct.NigeCount);
    w.i32(P_CT_OONIGE_COUNT.param(), ct.OonigeCount);
    w.f32(P_CT_DISTANCE_GAP1.param(), ct.DistanceGap1);
    w.f32(P_CT_LANE_GAP1.param(), ct.LaneGap1);
    w.f32(P_CT_DISTANCE_GAP2.param(), ct.DistanceGap2);
    w.f32(P_CT_LANE_GAP2.param(), ct.LaneGap2);
    w.f32(P_CT_TIME_COEF1.param(), ct.TimeCoef1);
    w.f32(P_CT_TIME_COEF2.param(), ct.TimeCoef2);
    w.f32(P_CT_TIME_COEF3.param(), ct.TimeCoef3);
    let cp = &p.ConservePower;
    w.f32(P_CP_DECREASE_COEF_COMPETE_TOP.param(), cp.ConservePowerDecreaseCoefArray.iter().find(|x| x.Mode == 3).map_or(1.0, |x| x.Coef));
    w.f32(
        P_CP_DECREASE_ACCEL_COEF_COMPETE_TOP.param(),
        cp.ConservePowerDecreaseAccelCoefArray.iter().find(|x| x.Mode == 3).map_or(1.0, |x| x.Coef),
    );
    let cf = &p.CompeteFight;
    w.f32(P_CF_TARGET_CONTINUE_TIME.param(), cf.TargetContinueTime);
    w.f32(P_CF_SPEED_GAP.param(), cf.SpeedGap);
    w.i32(P_CF_TARGET_ORDER_PER.param(), cf.TargetOrderPer);
    w.f32(P_CF_TARGET_CONTINUE_DISTANCE.param(), cf.TargetContinueDistance);
    w.i32(P_CF_HP_PER.param(), cf.HpPer);
    w.i32(P_CF_HP_PER2.param(), cf.HpPer2);
    let pk = &p.PositionKeep;
    w.i32(P_PK_END_SECTION.param(), pk.PositionKeepEndSection);
    w.f32(P_PK_PACEMAKER_CHANGE_FORWARD_RANGE.param(), pk.PacemakerChangeForwardRunningStyleRange);
    w.f32(P_PK_PACEMAKER_CHANGE_NOT_MOST_FORWARD_COUNT.param(), pk.PacemakerChangeTopHorseNotMostForwardStyleCount);
    pack_params_update_horses(&mut w, p);
    w
}

fn pack_params_update_horses(w: &mut ParamWords, p: &ParamDefine) {
    w.f32(P_START_SPEED.param(), p.StartSpeed);
    w.f32(P_MIN_SPEED_GUTS_COEF_SQRT.param(), p.Speed.MinSpeedGutsCoefSqrt);
    w.f32(P_MIN_SPEED_GUTS_COEF.param(), p.Speed.MinSpeedGutsCoef);
    w.f32(P_TARGET_SPEED_MIN.param(), p.Speed.TargetSpeedMin);
    w.f32(P_ACCEL_POW_COEF.param(), p.accelPowCoef);
    w.f32(P_ACCEL_POW_COEF_UP_SLOPE.param(), p.accelPowCoefUpSlope);
    w.f32(P_ACCEL_POW_COEF_SQRT.param(), p.AccelPowCoefSqrt);
    w.f32(P_START_ACCEL_ADD.param(), p.StartAccelAdd);
    w.f32(P_DECL_RATE_START.param(), p.declRateStart);
    w.f32(P_DECL_RATE_MIDDLE.param(), p.declRateMiddle);
    w.f32(P_DECL_RATE_END.param(), p.declRateEnd);
    w.f32(P_DECL_RATE_PACE_DOWN.param(), p.DeclRatePositionKeepPaceDown);
    w.f32(P_DECL_RATE_HP_ZERO.param(), p.declRateHpZero);
    w.f32(P_DECL_BASE.param(), p.declBase);
    w.f32(P_FRONT_BLOCK_MAX_SPEED_RATE_MIN.param(), p.Block.FrontBlockMaxSpeedRateMin);
    w.f32(P_FRONT_BLOCK_MAX_SPEED_RATE_MAX.param(), p.Block.FrontBlockMaxSpeedRateMax);
    let sl = &p.Slope;
    w.f32(P_UP_SLOPE_ADD_SPEED_VAL1.param(), sl.UpSlopeAddSpeedVal1);
    w.f32(P_DOWN_SLOPE_ADD_SPEED_VAL1.param(), sl.DownSlopeAddSpeedVal1);
    w.f32(P_DOWN_SLOPE_ADD_SPEED_VAL2.param(), sl.DownSlopeAddSpeedVal2);
    w.f32(P_DOWN_SLOPE_START_WIZ_RATE.param(), sl.DownSlopeAddSpeedStartWizRate);
    w.f32(P_DOWN_SLOPE_END_PER.param(), sl.DownSlopeAddSpeedEndPer);
    w.f32(P_LANE_MOVE_ADD_PARAM1.param(), p.Skill.LaneMoveAddParam1);
    w.f32(P_LANE_MOVE_ADD_PARAM2.param(), p.Skill.LaneMoveAddParam2);
    let (ct, cf) = (&p.CompeteTop, &p.CompeteFight);
    w.f32(P_CT_ADD_PARAM1_COEF1.param(), ct.AddParam1Coef1);
    w.f32(P_CT_ADD_PARAM1_COEF2.param(), ct.AddParam1Coef2);
    w.f32(P_CT_ADD_PARAM1_COEF3.param(), ct.AddParam1Coef3);
    w.f32(P_CF_ADD_PARAM1_COEF1.param(), cf.AddParam1Coef1);
    w.f32(P_CF_ADD_PARAM1_COEF2.param(), cf.AddParam1Coef2);
    w.f32(P_CF_ADD_PARAM1_COEF3.param(), cf.AddParam1Coef3);
    w.f32(P_CF_ADD_PARAM2_COEF1.param(), cf.AddParam2Coef1);
    w.f32(P_CF_ADD_PARAM2_COEF2.param(), cf.AddParam2Coef2);
    w.f32(P_CF_ADD_PARAM2_COEF3.param(), cf.AddParam2Coef3);
    let hp = &p.Hp;
    w.f32(P_HP_SPEED_GAP_PARAM1.param(), hp.SpeedGapParam1);
    w.f32(P_HP_SPEED_GAP_PARAM1_POW.param(), hp.SpeedGapParam1Pow);
    w.f32(P_HP_DEC_BASE.param(), hp.HpDecBase);
    w.f32(P_HP_GUTS_COEF.param(), hp.HpGutsCoef);
    w.f32(P_HP_GUTS_COEF_SQRT.param(), hp.HpGutsCoefSqrt);
    w.f32(P_HP_RATE_NORMAL.param(), hp.HpDecRateBaseNormal);
    w.f32(P_HP_RATE_TEMPT_CT_OONIGE.param(), hp.HpDecRateBaseTemptationAndCompeteTopOonige);
    w.f32(P_HP_RATE_TEMPT_CT_NIGE.param(), hp.HpDecRateBaseTemptationAndCompeteTopNige);
    w.f32(P_HP_RATE_CT_OONIGE.param(), hp.HpDecRateBaseCompeteTopOonige);
    w.f32(P_HP_RATE_CT_NIGE.param(), hp.HpDecRateBaseCompeteTopNige);
    w.f32(P_HP_RATE_TEMPTATION.param(), hp.HpDecRateBaseTemptation);
    w.f32(P_HP_RATE_PACE_DOWN.param(), hp.HpDecRateBasePositionKeepPaceDown);
    w.f32(P_HP_RATE_DOWN_SLOPE_MULTIPLY.param(), hp.HpDecRateMultiplyDownSlopeAccelMode);
    w.f32(P_LS_BASE_TARGET_SPEED_ADD_COEF.param(), p.lastSpurtBaseTargetSpeedAddCoef);
    w.i32(P_LS_DISTANCE_GOAL_BUFFER.param(), p.LastSpurtDistanceGoalBuffer);
    w.f32(P_LS_SPEED_DELTA.param(), p.lastSpurtSpeedDelta);
    w.f32(P_LS_STEP2_SPEED_DELTA.param(), p.lastSpurtStep2SpeedDelta);
    w.f32(P_LS_STEP2_PER_VAL1.param(), p.lastSpurtStep2PerVal1);
    w.f32(P_LS_STEP2_PER_VAL2.param(), p.lastSpurtStep2PerVal2);
    w.f32(P_LS_GUTS_COEF1.param(), p.LastSpurtTargetSpeedGutsCoef1);
    w.f32(P_LS_GUTS_POW.param(), p.LastSpurtTargetSpeedGutsPow);
    w.f32(P_LS_GUTS_COEF2.param(), p.LastSpurtTargetSpeedGutsCoef2);
    w.f32(P_LS_COEF_SQRT.param(), p.LastSpurtTargetSpeedCoefSqrt);
    w.f32(P_ADD_SPEED_PARAM_COEF.param(), p.addSpeedParamCoef);
    w.f32(P_BASE_TARGET_SPEED_COEF.param(), p.baseTargetSpeedCoef);
    w.f32(P_LS_DISTANCE_MAX_FROM_GOAL.param(), p.lastSpurtDistanceMaxFromGoal);
    w.f32(P_FORCE_IN_MOVE_LANE_THRESHOLD.param(), p.forceInMoveLaneThreshold);
    w.f32(P_LANE_MOVE_SPEED_BASE.param(), p.laneMoveSpeedBase);
    w.f32(P_LANE_MOVE_SPEED_ADD.param(), p.laneMoveSpeedAdd);
    w.f32(P_LANE_MOVE_SPEED_POW_COEF.param(), p.laneMoveSpeedPowCoef);
    w.f32(P_LANE_MOVE_SPEED_OUT_COEF.param(), p.laneMoveSpeedOutCoef);
    w.f32(P_LANE_MOVE_ACCEL_COEF.param(), p.laneMoveAccelMoveSpeedBaseCoef);
    w.f32(P_LAST_MOVE_OUT_LANE_NORMALIZE.param(), p.lastMoveOutLaneNormalizeVal);
    w.f32(P_LAST_MOVE_OUT_LANE_COEF.param(), p.lastMoveOutLaneCoef);
    w.f32(P_LAST_MOVE_OUT_LANE_ADD_RANDOM.param(), p.lastMoveOutLaneAddRandom);
    w.f32(P_OVERTAKE_OUT_LANE_COEF_END.param(), p.overTakeOutLaneCoefEnd);
    w.f32(P_OVERTAKE_IN_LANE_COEF_END.param(), p.overTakeInLaneCoefEnd);
    w.f32(P_OVERTAKE_COOL_DOWN_TIME.param(), p.overTakeCoolDownTime);
    w.f32(P_OVERTAKE_DIST_PER_SPEED.param(), p.overTakeDistPerSpeed);
    let pk = &p.PositionKeep;
    w.i32(P_PK_COOL_DOWN_SEC.param(), pk.PositionKeepCoolDownSec);
    w.i32(P_PK_CHECK_INTERVAL_SEC.param(), pk.PositionKeepCheckIntervalSec);
    w.i32(P_PK_PACE_BASE_DISTANCE.param(), pk.PositionKeepPaceBaseDistance);
    w.f32(P_PK_PACE_BASE_DISTANCE_COEF.param(), pk.PositionKeepPaceBaseDistanceCoef);
    let mins = [0.0, 0.0, pk.PositionKeepPaceUpDownDistanceDiffMinSenko, pk.PositionKeepPaceUpDownDistanceDiffMinSashi, pk.PositionKeepPaceUpDownDistanceDiffMinOikomi];
    let maxs = [0.0, 0.0, pk.PositionKeepPaceUpDownDistanceDiffMaxSenko, pk.PositionKeepPaceUpDownDistanceDiffMaxSashi, pk.PositionKeepPaceUpDownDistanceDiffMaxOikomi];
    for k in 0..5 {
        w.f32(P_PK_DIFF_MIN.param_k(k), mins[k]);
        w.f32(P_PK_DIFF_MAX.param_k(k), maxs[k]);
    }
    w.f32(P_PK_DIFF_MAX_COEF.param(), pk.PositionKeepPaceUpDownDistanceDiffMaxCoef);
    w.f32(P_PK_PACE_DOWN_TARGET_LANE.param(), pk.PositionKeepPaceDownTargetLane);
    w.f32(P_PK_PACE_DOWN_MULT.param(), pk.PositionKeepPaceDownBaseTargetSpeedMultiply);
    w.f32(P_PK_PACE_DOWN_MULT_MIDDLE.param(), pk.PositionKeepPaceDownBaseTargetSpeedMultiplyPhaseMiddle);
    w.f32(P_PK_SPEED_UP_PER_VAL1.param(), pk.PositionKeepSpeedUpStartPerVal1);
    w.f32(P_PK_SPEED_UP_PER_VAL1_OONIGE.param(), pk.PositionKeepSpeedUpStartPerVal1Oonige);
    w.f32(P_PK_OVERTAKE_PER_VAL1.param(), pk.PositionKeepOvertakeStartPerVal1);
    w.f32(P_PK_OVERTAKE_PER_VAL1_OONIGE.param(), pk.PositionKeepOvertakeStartPerVal1Oonige);
    w.f32(P_PK_PACE_UP_PER_VAL1.param(), pk.PositionKeepPaceUpStartPerVal1);
    w.f32(P_PK_SPEED_UP_END_DIFF.param(), pk.PositionKeepSpeedUpEndDistanceDiff);
    w.f32(P_PK_SPEED_UP_END_DIFF_OONIGE.param(), pk.PositionKeepSpeedUpEndDistanceDiffOonige);
    w.f32(P_PK_SPEED_UP_END_DIFF_ONE.param(), pk.PositionKeepSpeedUpEndDistanceDiffOne);
    w.f32(P_PK_OVERTAKE_END_DIFF.param(), pk.PositionKeepOvertakeEndDistanceDiff);
    w.f32(P_PK_OVERTAKE_END_DIFF_OONIGE.param(), pk.PositionKeepOvertakeEndDistanceDiffOonige);
    let tm = &p.Temptation;
    w.f32(P_TM_END_CHECK_TIME.param(), tm.TemptationEndCheckTime);
    w.f32(P_TM_FORCE_END_TIME.param(), tm.TemptationForceEndTime);
    w.f32(P_TM_END_PER.param(), tm.TemptationEndPer);
    w.f32(P_TM_SENKO_TO_NIGE.param(), tm.SenkoToNigePer);
    w.f32(P_TM_SASHI_TO_SENKO.param(), tm.SashiToSenkoPer);
    w.f32(P_TM_SASHI_TO_NIGE.param(), tm.SashiToNigePer);
    w.f32(P_TM_OIKOMI_TO_SASHI.param(), tm.OikomiToSashiPer);
    w.f32(P_TM_OIKOMI_TO_SENKO.param(), tm.OikomiToSenkoPer);
    w.f32(P_TM_OIKOMI_TO_NIGE.param(), tm.OikomiToNigePer);
    let cp = &p.ConservePower;
    let by_mode = |a: &[(i32, f32)], m: i32| a.iter().find(|x| x.0 == m).map_or(1.0, |x| x.1);
    let dec: Vec<_> = cp.ConservePowerDecreaseCoefArray.iter().map(|x| (x.Mode, x.Coef)).collect();
    let dec_accel: Vec<_> = cp.ConservePowerDecreaseAccelCoefArray.iter().map(|x| (x.Mode, x.Coef)).collect();
    let inc: Vec<_> = cp.ConservePowerIncreaseCoefArray.iter().map(|x| (x.Mode, x.Coef)).collect();
    let style: Vec<_> = cp.ConservePowerRunningStyleCoefArray.iter().map(|x| (x.RunningStyle, x.Coef)).collect();
    let dtype: Vec<_> = cp.ReleaseConservePowerActivityTimeDistanceTypeCoefArray.iter().map(|x| (x.DistanceType, x.Coef)).collect();
    w.f32(P_CP_DECREASE_COEF_TEMPTATION.param(), by_mode(&dec, 4));
    w.f32(P_CP_DECREASE_ACCEL_COEF_TEMPTATION.param(), by_mode(&dec_accel, 4));
    w.f32(P_CP_INCREASE_COEF_PACE_DOWN.param(), by_mode(&inc, 1));
    w.f32(P_CP_INCREASE_COEF_NORMAL.param(), by_mode(&inc, 2));
    for k in 0..5 {
        w.f32(P_CP_STYLE_COEF.param_k(k), by_mode(&style, k as i32));
        w.f32(P_CP_RELEASE_TIME_DISTANCE_COEF.param_k(k), by_mode(&dtype, k as i32));
        for t in 0..5 {
            let a = &cp.ReleaseConservePowerAccelRunningStyleAndDistanceTypeCoefArray;
            let c = a.iter().find(|x| x.RunningStyle == k as i32 && x.DistanceType == t as i32).map_or(1.0, |x| x.Coef);
            w.f32(P_CP_RELEASE_ACCEL_COEF_BY.param_k(k * 5 + t), c);
        }
    }
    w.f32(P_CP_RELEASE_TIME_COEF.param(), cp.ReleaseConservePowerActivityTimeCoef);
    w.f32(P_CP_RELEASABLE_TIME_THRESHOLD.param(), cp.ConservePowerReleasableActivityTimeThreshold);
    w.f32(P_CP_RELEASE_DECL_COEF.param(), cp.ReleaseConservePowerDeclCoef);
    w.f32(P_CP_RELEASE_ACCEL_COEF.param(), cp.ReleaseConservePowerAccelCoef);
    w.f32(P_CP_DEFAULT_COOL_DOWN.param(), cp.ConservePowerDefaultCoolDownTime);
    w.f32(P_FINISH_TIME_COEF.param(), p.Global.FinishTimeCoef);
    w.f32(P_LOT_ACTIVATE_PER_VAL1.param(), p.Skill.LotActivatePerVal1);
    w.f32(P_LOT_ACTIVATE_PER_VAL2.param(), p.Skill.LotActivatePerVal2);
    w.i32(P_ACTIVATE_PER_MIN.param(), p.Skill.ActivatePerMin);
}

fn modifier_flags(m: &SkillParamModifier) -> u32 {
    (m.kind == SkillParamModifierKind::Set) as u32 | (m.is_one_time as u32) << 1 | (m.is_time_infinite as u32) << 2
}

/// The receiver's lists as one pool ordered by parameter (each list keeps its order), and the one-time list.
fn pack_modifiers(s: &mut LaneState, h: usize, r: &SkillModifierReceiver) {
    let mut pool: Vec<(i32, &SkillParamModifier)> = Vec::new();
    for t in 0..26 {
        for (param, list) in &r.modifier {
            if *param as i32 == t {
                pool.extend(list.iter().map(|m| (t, m)));
            }
        }
    }
    assert!(pool.len() <= MAX_MODIFIERS as usize, "{} modifiers: raise MAX_MODIFIERS", pool.len());
    s.i32(H_MOD_COUNT.horse(h), pool.len() as i32);
    for k in 0..MAX_MODIFIERS as usize {
        let e = pool.get(k);
        s.i32(H_MOD_PARAM.horse_k(h, k), e.map_or(0, |e| e.0));
        s.u32(H_MOD_FLAGS.horse_k(h, k), e.map_or(0, |e| modifier_flags(e.1)));
        s.f32(H_MOD_TIME.horse_k(h, k), e.map_or(0.0, |e| e.1.time));
        s.f32(H_MOD_VALUE.horse_k(h, k), e.map_or(0.0, |e| e.1.value));
        s.u32(H_MOD_ID_LO.horse_k(h, k), e.map_or(0, |e| e.1.id as u32));
        s.u32(H_MOD_ID_HI.horse_k(h, k), e.map_or(0, |e| (e.1.id >> 32) as u32));
    }
    let mut once: Vec<&SkillParamModifier> = Vec::new();
    for (param, list) in &r.one_time_skill_modifier {
        assert!(list.is_empty() || *param == SkillModifierParam::CurrentSpeedWithNaturalDeceleration, "one-time {param:?}");
        once.extend(list.iter());
    }
    assert!(once.len() <= MAX_ONE_TIME_MODIFIERS as usize, "raise MAX_ONE_TIME_MODIFIERS");
    s.i32(H_OTM_COUNT.horse(h), once.len() as i32);
    for k in 0..MAX_ONE_TIME_MODIFIERS as usize {
        let e = once.get(k);
        s.u32(H_OTM_FLAGS.horse_k(h, k), e.map_or(0, |m| modifier_flags(m)));
        s.f32(H_OTM_TIME.horse_k(h, k), e.map_or(0.0, |m| m.time));
        s.f32(H_OTM_VALUE.horse_k(h, k), e.map_or(0.0, |m| m.value));
        s.u32(H_OTM_ID_LO.horse_k(h, k), e.map_or(0, |m| m.id as u32));
        s.u32(H_OTM_ID_HI.horse_k(h, k), e.map_or(0, |m| (m.id >> 32) as u32));
    }
}
