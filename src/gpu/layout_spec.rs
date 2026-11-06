// The GPU race state: every field a ported stage reads or writes. Plain data only, because build.rs `include!`s this
// file to generate shaders' race_state.glsl and the plane constants in `gpu::layout` from the same list.
//
// Storage: one u32 "plane" per value, `lanes` words long, lane (= race) fastest, so a wave reading one field of one
// horse touches consecutive words. A horse field of length `len` has len * MAX_HORSES planes (element-major), an
// f64 two planes per value (low word first). GLSL accessors: h_<name>(h[, k]) / set_h_<name>(h, [k,] v) for horse
// fields, r_<name>([k]) / set_r_<name>([k,] v) for race fields.
//
// Param fields (race_paramdefine values, the same for every race of a data era) are not per lane: they live in a
// separate read-only buffer, one word per value, read with p_<name>([k]).
//
// To add a field: append it here, write it in gpu::pack, then use it in GLSL. gpu_stagecheck fails if pack misses one.

#[allow(dead_code)]
pub const MAX_HORSES: u32 = 18;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    F32,
    I32,
    U32,
    Bool,
    F64,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Race,
    Horse,
    Param,
}

#[allow(dead_code)]
pub struct FieldSpec {
    pub scope: Scope,
    pub name: &'static str,
    pub kind: Kind,
    pub len: u32,
}

const fn race(name: &'static str, kind: Kind, len: u32) -> FieldSpec {
    FieldSpec { scope: Scope::Race, name, kind, len }
}
const fn horse(name: &'static str, kind: Kind, len: u32) -> FieldSpec {
    FieldSpec { scope: Scope::Horse, name, kind, len }
}
const fn param(name: &'static str, kind: Kind, len: u32) -> FieldSpec {
    FieldSpec { scope: Scope::Param, name, kind, len }
}

/// Capacity of Skill.BehindNearParamArray (1 entry in the current data).
#[allow(dead_code)]
pub const MAX_BEHIND_NEAR_PARAMS: u32 = 4;
/// Capacity of a horse's modifier pool (active skill modifiers; 14 at most in 963 replays). A kernel that would
/// exceed it sets gpu_fault.
#[allow(dead_code)]
pub const MAX_MODIFIERS: u32 = 24;
/// Capacity of the one-time modifier list (CurrentSpeedWithNaturalDeceleration only).
#[allow(dead_code)]
pub const MAX_ONE_TIME_MODIFIERS: u32 = 8;
/// The race RNG (rng::NetRandom): seed array 0..56, inext, inextp, draw count low / high word.
#[allow(dead_code)]
pub const RNG_WORDS: u32 = 60;
/// Sections of a horse's base target speed table (speed::BaseTargetSpeeds).
#[allow(dead_code)]
pub const BTS_SECTIONS: u32 = 24;
/// Position-keep strategies, indexed by PositionKeepMode (0 Null .. 5 PaceUpEx).
#[allow(dead_code)]
pub const PK_STRATEGIES: u32 = 6;
/// Order-in / order-out condition types (10% ..= 90%).
#[allow(dead_code)]
pub const ORDER_IN_OUT_TYPES: u32 = 9;

#[allow(dead_code)]
#[rustfmt::skip]
pub const FIELDS: &[FieldSpec] = &[
    // ---- race
    race("num_horses", Kind::I32, 1),
    // Set by a kernel that reaches a path it does not implement (a GPU-only field; the CPU always packs 0).
    race("gpu_fault", Kind::U32, 1),
    // field.standings.horse_array_by_order (entries past num_horses are 0).
    race("order_horse", Kind::I32, MAX_HORSES),
    race("order_distance", Kind::F32, MAX_HORSES),
    // Set by UpdateHorses: every horse has finished (the CPU's frame-local all_finished). pack writes false;
    // gpu_stagecheck sets it.
    race("frame_all_finished", Kind::Bool, 1),
    race("num_race_horses", Kind::I32, 1),
    race("last_order", Kind::I32, 1),
    // Finish order (last frame).
    race("is_finish_order_calculated", Kind::Bool, 1),
    race("is_defeat_checked", Kind::Bool, 1),
    race("finish_order_horse", Kind::I32, MAX_HORSES),
    // calc_clamped_finish_time: the course set's bounds, and the clamped time for each side (a fresh
    // RaceRandom(seed)'s first draw, computed on the CPU).
    race("finish_time_min", Kind::F32, 1),
    race("finish_time_max", Kind::F32, 1),
    race("finish_time_clamp_low", Kind::F32, 1),
    race("finish_time_clamp_high", Kind::F32, 1),
    // Per running style (index = RunningStyle, 0 unused): the style's horses in current order, and the cached ends
    // (-1 = none).
    race("style_count", Kind::I32, 5),
    race("style_len", Kind::I32, 5),
    race("style_horse", Kind::I32, 5 * MAX_HORSES),
    race("style_top_horse", Kind::I32, 5),
    race("style_last_horse", Kind::I32, 5),
    race("rushed_len", Kind::I32, 1),
    race("rushed_horse", Kind::I32, MAX_HORSES),
    race("style_rushed_len", Kind::I32, 5),
    race("style_rushed_horse", Kind::I32, 5 * MAX_HORSES),
    // field.scan_view: distance, lane and last speed of every horse at the start of PreUpdate, and a bit per horse
    // on the final straight.
    race("sv_distance", Kind::F32, MAX_HORSES),
    race("sv_lane", Kind::F32, MAX_HORSES),
    race("sv_last_speed", Kind::F32, MAX_HORSES),
    race("sv_straight_last", Kind::U32, 1),
    // Field
    race("course_section_distance", Kind::F32, 1),
    race("compete_top_start_count", Kind::I32, 2),
    race("most_forward_running_style", Kind::I32, 1),
    // field.pacemaker (-1 = none)
    race("pm_horse", Kind::I32, 1),
    race("pm_section", Kind::I32, 1),
    race("pm_section_pick", Kind::I32, 1),
    race("pm_not_most_forward_count", Kind::I32, 1),
    race("pm_overriding", Kind::Bool, 1),
    // UpdateHorses
    race("rng", Kind::U32, RNG_WORDS),
    race("accumulate_time", Kind::F32, 1),
    race("world_per_metre", Kind::F32, 1),
    race("base_speed", Kind::F32, 1),
    // race.info.phase_distances
    race("phase_middle_start", Kind::F32, 1),
    race("phase_end_start", Kind::F32, 1),
    race("phase_last_start", Kind::F32, 1),
    race("phase_course_distance", Kind::F32, 1),
    race("course_distance", Kind::I32, 1),
    race("course_distance_type", Kind::I32, 1),
    race("info_lane_distance_max", Kind::F32, 1),
    race("first_move_lane_point", Kind::F32, 1),
    // data.era.pre_2026_07_courses
    race("pre_2026_07", Kind::Bool, 1),
    // Word offset of this race's course path in the course buffer (gpu::pack::course_words). gpu_stagecheck sets
    // it when it builds a batch; pack writes 0.
    race("course_base", Kind::U32, 1),
    // Skills: word offset of this race's block in the skill program buffer (gpu::skills; set like course_base), and
    // race.next_modifier_id (low, high word)
    race("skill_base", Kind::U32, 1),
    race("next_modifier_id", Kind::U32, 2),
    // tick::end_step: race.distance_diff_max (a result), info.race_type == STORY (lane-max events are ignored)
    race("distance_diff_max", Kind::F32, 1),
    race("is_story", Kind::Bool, 1),
    // Set by the end_step kernel on the race's last step; every kernel then returns at once. gpu::pack writes false.
    race("race_done", Kind::Bool, 1),

    // ---- horse
    horse("distance", Kind::F32, 1),
    horse("cur_order", Kind::I32, 1),
    horse("prev_order", Kind::I32, 1),
    horse("is_finished", Kind::Bool, 1),
    horse("phase", Kind::I32, 1),
    horse("cur_corner", Kind::I32, 1),
    horse("is_final_corner", Kind::Bool, 1),
    // ai.last_spurt_start_distance, ai.last_spurt_calc_result
    horse("ls_start_distance", Kind::F32, 1),
    horse("ls_calc_result", Kind::I32, 1),
    // overtake_target_have_no_order_{up,down}_continue_time
    horse("no_order_up_time", Kind::F32, 1),
    horse("no_order_down_time", Kind::F32, 1),
    // order_change_counter_up (ocu) and order_change_counter_down (ocd)
    horse("ocu_cooldown", Kind::F32, MAX_HORSES),
    horse("ocu_cool_down_time", Kind::F32, 1),
    horse("ocu_phase_start", Kind::I32, 1),
    horse("ocu_phase_middle", Kind::I32, 1),
    horse("ocu_phase_end_after", Kind::I32, 1),
    horse("ocu_corner", Kind::I32, 1),
    horse("ocu_corner_phase_end_after", Kind::I32, 1),
    horse("ocu_last_spurt", Kind::I32, 1),
    horse("ocu_distance1", Kind::I32, 1),
    horse("ocu_final_corner_after", Kind::I32, 1),
    horse("ocu_changed", Kind::Bool, 1),
    horse("ocd_cooldown", Kind::F32, MAX_HORSES),
    horse("ocd_cool_down_time", Kind::F32, 1),
    horse("ocd_phase_start", Kind::I32, 1),
    horse("ocd_phase_middle", Kind::I32, 1),
    horse("ocd_phase_end_after", Kind::I32, 1),
    horse("ocd_corner", Kind::I32, 1),
    horse("ocd_corner_phase_end_after", Kind::I32, 1),
    horse("ocd_last_spurt", Kind::I32, 1),
    horse("ocd_distance1", Kind::I32, 1),
    horse("ocd_final_corner_after", Kind::I32, 1),
    horse("ocd_changed", Kind::Bool, 1),
    horse("lane_distance", Kind::F32, 1),
    horse("lane_distance_per_in_all_horses", Kind::F32, 1),
    horse("last_order_time", Kind::F32, 1),
    horse("is_temptation", Kind::Bool, 1),
    // Near-horse timers (PostUpdate).
    horse("in_front_near_time", Kind::F32, 1),
    horse("in_front_near_lane_time", Kind::F32, 1),
    horse("behind_near_time", Kind::F32, 1),
    horse("behind_near_lane_time", Kind::F32, 1),
    horse("behind_near_lane_times", Kind::F32, MAX_BEHIND_NEAR_PARAMS),
    // Finish order and defeat reason.
    horse("finish_time_scaled", Kind::F32, 1),
    horse("finish_time_for_order", Kind::F64, 1),
    horse("finish_order", Kind::I32, 1),
    horse("finish_time_diff_from_prev", Kind::F32, 1),
    horse("defeat", Kind::I32, 1),
    horse("running_style", Kind::I32, 1),
    horse("motivation", Kind::I32, 1),
    horse("active_proper_distance", Kind::I32, 1),
    horse("active_proper_ground_type", Kind::I32, 1),
    horse("is_hp_empty_on_race", Kind::Bool, 1),
    horse("temptation_count", Kind::I32, 1),
    horse("guts_order", Kind::I32, 1),
    horse("skill_count", Kind::I32, 1),
    horse("block_front_accumulate_time", Kind::F32, 1),
    // PreUpdate
    horse("last_speed", Kind::F32, 1),
    horse("is_straight", Kind::Bool, 1),
    horse("lane_distance_max", Kind::F32, 1),
    horse("corner_end_distance", Kind::F32, 1),
    horse("move_distance", Kind::F32, 1),
    horse("base_visible_distance", Kind::F32, 1),
    horse("after_goal_time", Kind::F32, 1),
    // ActiveLanePlanner: 0 Race, 1 AfterGoal
    horse("active_lane_planner", Kind::I32, 1),
    // ai.around (AroundTable), indexed by the other horse
    horse("ar_distance_gap", Kind::F32, MAX_HORSES),
    horse("ar_distance_gap_abs", Kind::F32, MAX_HORSES),
    horse("ar_lane_gap", Kind::F32, MAX_HORSES),
    horse("ar_lane_gap_abs", Kind::F32, MAX_HORSES),
    horse("ar_speed_gap", Kind::F32, MAX_HORSES),
    horse("ar_speed_gap_abs", Kind::F32, MAX_HORSES),
    horse("ar_cf_time", Kind::F32, MAX_HORSES),
    horse("ar_back", Kind::U32, 1),
    horse("ar_others", Kind::U32, 1),
    // ai.near_horses / ai.visibility.visible_horses: only the first *_count entries are current (the rest are
    // whatever an earlier frame left).
    horse("near_horses", Kind::I32, MAX_HORSES),
    horse("near_horse_count", Kind::I32, 1),
    horse("visible_horses", Kind::I32, MAX_HORSES),
    horse("visible_horse_count", Kind::I32, 1),
    // compete_fight_near (a list; entries past cf_near_len are 0)
    horse("cf_near_len", Kind::I32, 1),
    horse("cf_near", Kind::I32, MAX_HORSES),
    // ai.block_horse / _in / _out (-1 = none)
    horse("block_horse", Kind::I32, 1),
    horse("block_horse_in", Kind::I32, 1),
    horse("block_horse_out", Kind::I32, 1),
    horse("in_lane_space", Kind::F32, 1),
    horse("out_lane_space", Kind::F32, 1),
    horse("is_around_horse_initialized", Kind::Bool, 1),
    horse("is_surround_out", Kind::Bool, 1),
    horse("is_surround_ahead", Kind::Bool, 1),
    horse("is_surround_behind", Kind::Bool, 1),
    // Field
    horse("running_style_ex", Kind::I32, 1),
    horse("hp", Kind::F32, 1),
    horse("max_hp", Kind::F32, 1),
    horse("base_guts_adjusted", Kind::F32, 1),
    horse("proper_running_style_coef", Kind::F32, 1),
    horse("is_compete_top", Kind::Bool, 1),
    horse("compete_top_count", Kind::I32, 1),
    horse("compete_top_group", Kind::I32, 1),
    horse("compete_top_remain_time", Kind::F32, 1),
    horse("is_compete_fight", Kind::Bool, 1),
    horse("compete_fight_count", Kind::I32, 1),
    horse("compete_fight_group", Kind::I32, 1),
    // ai.conserve_power
    horse("conserved_power", Kind::F32, 1),
    horse("conserve_activity_coef", Kind::F32, 1),
    // is_force_check_next_lane_one_time of ai.lane_planner (either variant) and of ai.after_goal_lane
    horse("lp_force_check", Kind::Bool, 1),
    horse("agl_force_check", Kind::Bool, 1),
    // skill_modifier_receiver.modifier: every list in one pool, ordered by parameter (SkillModifierParam), each
    // parameter's entries in their list order (the only order apply_modifier and update_modifiers depend on).
    // flags: 1 Set (else Add), 2 one-time, 4 time-infinite. Entries past mod_count are 0.
    horse("mod_count", Kind::I32, 1),
    horse("mod_param", Kind::I32, MAX_MODIFIERS),
    horse("mod_flags", Kind::U32, MAX_MODIFIERS),
    horse("mod_time", Kind::F32, MAX_MODIFIERS),
    horse("mod_value", Kind::F32, MAX_MODIFIERS),
    horse("mod_id_lo", Kind::U32, MAX_MODIFIERS),
    horse("mod_id_hi", Kind::U32, MAX_MODIFIERS),
    // skill_modifier_receiver.one_time_skill_modifier (only CurrentSpeedWithNaturalDeceleration ever has entries)
    horse("otm_count", Kind::I32, 1),
    horse("otm_flags", Kind::U32, MAX_ONE_TIME_MODIFIERS),
    horse("otm_time", Kind::F32, MAX_ONE_TIME_MODIFIERS),
    horse("otm_value", Kind::F32, MAX_ONE_TIME_MODIFIERS),
    horse("otm_id_lo", Kind::U32, MAX_ONE_TIME_MODIFIERS),
    horse("otm_id_hi", Kind::U32, MAX_ONE_TIME_MODIFIERS),
    // UpdateHorses: the horse's fixed values
    horse("race_default_min_speed", Kind::F32, 1),
    horse("base_speed_adjusted", Kind::F32, 1),
    horse("base_pow_adjusted", Kind::F32, 1),
    horse("base_wiz_adjusted", Kind::F32, 1),
    horse("base_wiz_adjusted_alt", Kind::F32, 1),
    horse("wiz_alt_mechanics", Kind::U32, 1),
    horse("multi_hp_sub", Kind::F32, 1),
    horse("proper_distance_coef_speed", Kind::F32, 1),
    horse("proper_distance_coef_pow", Kind::F32, 1),
    horse("proper_ground_type_coef", Kind::F32, 1),
    // accel.phase_accel_coef: start, middle, end, last
    horse("phase_accel_coef", Kind::F32, 4),
    horse("is_empty_in_move", Kind::Bool, 1),
    // ... and its live state
    horse("is_start_dash", Kind::Bool, 1),
    horse("min_speed", Kind::F32, 1),
    horse("delay_time", Kind::F32, 1),
    horse("last_self_speed", Kind::F32, 1),
    horse("lane_move_speed", Kind::F32, 1),
    horse("last_lane_move_speed", Kind::F32, 1),
    horse("move_lane_continue_time", Kind::F32, 1),
    // LaneDirection: 0 In, 1 Out
    horse("move_lane_dir", Kind::I32, 1),
    horse("prev_lane_distance", Kind::F32, 1),
    horse("lane_distance_min", Kind::F32, 1),
    horse("prev_distance", Kind::F32, 1),
    horse("prev_elapsed_time", Kind::F32, 1),
    horse("position", Kind::F32, 3),
    horse("prev_position", Kind::F32, 3),
    horse("slope_type", Kind::I32, 1),
    horse("slope_per", Kind::F32, 1),
    horse("slope_end_distance", Kind::F32, 1),
    horse("last_spurt_order", Kind::I32, 1),
    horse("final_corner_end_order", Kind::I32, 1),
    horse("is_last_spurt_prev", Kind::Bool, 1),
    horse("block_front_continue_time", Kind::F32, 1),
    horse("block_side_continue_time", Kind::F32, 1),
    horse("block_front_hp_empty_time", Kind::F32, 1),
    horse("block_side_max_by_phase", Kind::F32, 4),
    horse("near_around_time", Kind::F32, 1),
    horse("overtake_target_time", Kind::F32, 1),
    horse("position_keep_count", Kind::I32, 1),
    horse("order_in_continue", Kind::Bool, ORDER_IN_OUT_TYPES),
    horse("order_out_continue", Kind::Bool, ORDER_IN_OUT_TYPES),
    horse("finish_time_raw", Kind::F32, 1),
    // ai
    horse("target_speed", Kind::F32, 1),
    horse("target_lane", Kind::F32, 1),
    horse("down_slope_mode", Kind::Bool, 1),
    horse("down_slope_check_sec", Kind::F32, 1),
    horse("req_calc_ls", Kind::Bool, 1),
    horse("ls_by_distance_requested", Kind::Bool, 1),
    horse("force_in_move", Kind::Bool, 1),
    horse("ls_calc_count", Kind::I32, 1),
    horse("ls_target_speed", Kind::F32, 1),
    // ai.target_speed_calc
    horse("force_in_move_add", Kind::F32, 1),
    horse("calc_target_speed", Kind::F32, 1),
    // ai.overtake_targets (a list in insertion order; entries past ot_count are 0) and the cooldown
    horse("ot_count", Kind::I32, 1),
    horse("ot_targets", Kind::I32, MAX_HORSES),
    horse("ot_cool_down", Kind::F32, 1),
    // ai.base_target_speeds (Phase per section)
    horse("bts_initialized", Kind::Bool, 1),
    horse("bts_speed", Kind::F32, BTS_SECTIONS),
    horse("bts_random", Kind::F32, BTS_SECTIONS),
    horse("bts_phase", Kind::I32, BTS_SECTIONS),
    // ai.position_keep. The keep rule is Nige { is_force_speed_up } when pk_style is Nige, else PaceAdjust; the
    // pace strategies (PaceUp, PaceDown, PaceUpEx = 3, 4, 5) always hold the same style as the rule.
    horse("pk_check_start", Kind::F32, 1),
    horse("pk_check_end", Kind::F32, 1),
    horse("pk_style", Kind::I32, 1),
    horse("pk_force_speed_up", Kind::Bool, 1),
    horse("pk_temptation_prev", Kind::Bool, 1),
    horse("pk_cur", Kind::I32, 1),
    // continue_distance of strategies 1..=5 (strategy 0 is the default one, never started)
    horse("pk_continue_distance", Kind::F32, 1),
    horse("pk_end_distance", Kind::F32, PK_STRATEGIES),
    horse("pk_mult", Kind::F32, PK_STRATEGIES),
    // PaceState of strategies 3, 4, 5
    horse("pk_ps_min", Kind::F32, 3),
    horse("pk_ps_max", Kind::F32, 3),
    horse("pk_ps_target", Kind::F32, 3),
    horse("pk_cool_down", Kind::F32, 1),
    horse("pk_check_interval", Kind::F32, 1),
    // ai.temptation (is_temptation and temptation_count are above)
    horse("tm_start_section", Kind::I32, 1),
    horse("tm_end_check_sec", Kind::F32, 1),
    horse("tm_force_end_sec", Kind::F32, 1),
    horse("tm_skill_end_enable", Kind::Bool, 1),
    horse("tm_skill_end_time", Kind::F32, 1),
    horse("tm_mode", Kind::I32, 1),
    horse("tm_start_enable", Kind::Bool, 1),
    // ai.lane_planner: the Race variant's state (lp_is_race false = the Fixed variant, which holds only
    // lp_force_check)
    horse("lp_is_race", Kind::Bool, 1),
    horse("lp_last_move_out_start", Kind::F32, 1),
    horse("lp_extra_checked", Kind::Bool, 1),
    horse("lp_use_extra", Kind::Bool, 1),
    horse("lp_extra_lane", Kind::F32, 1),
    horse("lp_target_lane", Kind::F32, 1),
    // ai.conserve_power (conserved_power and activity_coef are above)
    horse("cp_active", Kind::Bool, 1),
    horse("cp_power_stat", Kind::F32, 1),
    horse("cp_cool_down", Kind::F32, 1),
    horse("cp_bank_timer", Kind::F32, 1),
    horse("cp_temptation_prev", Kind::Bool, 1),
    horse("cp_released", Kind::Bool, 1),
    horse("cp_accel", Kind::F32, 1),
    horse("cp_remain_time", Kind::F32, 1),
    // Skills
    horse("is_good_start", Kind::Bool, 1),
    horse("is_bad_start", Kind::Bool, 1),
    horse("cur_straight_front_type", Kind::I32, 1),
    horse("is_param_initialized", Kind::Bool, 1),
    horse("is_skill_initialized", Kind::Bool, 1),
    horse("is_complete_first_update", Kind::Bool, 1),
    horse("act_count_by_phase", Kind::I32, 4),
    horse("act_count_by_distance", Kind::I32, 2),
    horse("act_heal_count", Kind::I32, 1),
    horse("act_tag_group_count", Kind::I32, 1),
    // tick::end_step (course events): event_manager.last_event_by_horse, and fields only the events write
    horse("last_event", Kind::I32, 1),
    horse("ground_type", Kind::I32, 1),
    horse("cur_straight_type_number", Kind::I32, 1),
    horse("last_straight_order", Kind::I32, 1),

    // ---- params
    param("infront_near_distance", Kind::F32, 1),
    param("infront_near_lane_distance", Kind::F32, 1),
    param("behind_near_distance", Kind::F32, 1),
    param("behind_near_lane_distance", Kind::F32, 1),
    param("behind_near_param_count", Kind::I32, 1),
    param("behind_near_param_distance", Kind::F32, MAX_BEHIND_NEAR_PARAMS),
    param("behind_near_param_lane_distance", Kind::F32, MAX_BEHIND_NEAR_PARAMS),
    param("defeat_same_running_style_per", Kind::I32, 1),
    param("defeat_guts_order_per", Kind::I32, 1),
    param("defeat_temptation_count", Kind::I32, 1),
    param("defeat_front_block_accumulate_time", Kind::F32, 1),
    param("defeat_lose_finish_order", Kind::I32, 1),
    param("defeat_lose_bashin_diff", Kind::I32, 1),
    param("defeat_proper_distance", Kind::I32, 1),
    param("defeat_proper_ground", Kind::I32, 1),
    param("defeat_motivation", Kind::I32, 1),
    // Around-horse scan limits
    param("surround_side_distance", Kind::F32, 1),
    param("surround_side_lane", Kind::F32, 1),
    param("surround_distance", Kind::F32, 1),
    param("surround_lane", Kind::F32, 1),
    param("near_distance", Kind::F32, 1),
    param("near_lane", Kind::F32, 1),
    param("compete_fight_lane_gap", Kind::F32, 1),
    param("compete_fight_distance_gap", Kind::F32, 1),
    param("side_block_distance_gap", Kind::F32, 1),
    param("side_block_lane_gap", Kind::F32, 1),
    param("front_block_distance_gap", Kind::F32, 1),
    param("front_block_lane_gap", Kind::F32, 1),
    // CompeteTop (spot struggle)
    param("ct_check_start_distance", Kind::F32, 1),
    param("ct_check_end_section", Kind::I32, 1),
    param("ct_end_section", Kind::I32, 1),
    param("ct_nige_count", Kind::I32, 1),
    param("ct_oonige_count", Kind::I32, 1),
    param("ct_distance_gap1", Kind::F32, 1),
    param("ct_lane_gap1", Kind::F32, 1),
    param("ct_distance_gap2", Kind::F32, 1),
    param("ct_lane_gap2", Kind::F32, 1),
    param("ct_time_coef1", Kind::F32, 1),
    param("ct_time_coef2", Kind::F32, 1),
    param("ct_time_coef3", Kind::F32, 1),
    // ConservePower Mode 3 (spot struggle) decrease coefficients (1 when the arrays have no Mode 3 entry)
    param("cp_decrease_coef_compete_top", Kind::F32, 1),
    param("cp_decrease_accel_coef_compete_top", Kind::F32, 1),
    // CompeteFight (dueling)
    param("cf_target_continue_time", Kind::F32, 1),
    param("cf_speed_gap", Kind::F32, 1),
    param("cf_target_order_per", Kind::I32, 1),
    param("cf_target_continue_distance", Kind::F32, 1),
    param("cf_hp_per", Kind::I32, 1),
    param("cf_hp_per2", Kind::I32, 1),
    // PositionKeep (pacemaker)
    param("pk_end_section", Kind::I32, 1),
    param("pk_pacemaker_change_forward_range", Kind::F32, 1),
    param("pk_pacemaker_change_not_most_forward_count", Kind::F32, 1),
    // UpdateHorses: speed and acceleration
    param("start_speed", Kind::F32, 1),
    param("min_speed_guts_coef_sqrt", Kind::F32, 1),
    param("min_speed_guts_coef", Kind::F32, 1),
    param("target_speed_min", Kind::F32, 1),
    param("accel_pow_coef", Kind::F32, 1),
    param("accel_pow_coef_up_slope", Kind::F32, 1),
    param("accel_pow_coef_sqrt", Kind::F32, 1),
    param("start_accel_add", Kind::F32, 1),
    param("decl_rate_start", Kind::F32, 1),
    param("decl_rate_middle", Kind::F32, 1),
    param("decl_rate_end", Kind::F32, 1),
    param("decl_rate_pace_down", Kind::F32, 1),
    param("decl_rate_hp_zero", Kind::F32, 1),
    param("decl_base", Kind::F32, 1),
    param("front_block_max_speed_rate_min", Kind::F32, 1),
    param("front_block_max_speed_rate_max", Kind::F32, 1),
    param("up_slope_add_speed_val1", Kind::F32, 1),
    param("down_slope_add_speed_val1", Kind::F32, 1),
    param("down_slope_add_speed_val2", Kind::F32, 1),
    param("down_slope_start_wiz_rate", Kind::F32, 1),
    param("down_slope_end_per", Kind::F32, 1),
    param("lane_move_add_param1", Kind::F32, 1),
    param("lane_move_add_param2", Kind::F32, 1),
    param("ct_add_param1_coef1", Kind::F32, 1),
    param("ct_add_param1_coef2", Kind::F32, 1),
    param("ct_add_param1_coef3", Kind::F32, 1),
    param("cf_add_param1_coef1", Kind::F32, 1),
    param("cf_add_param1_coef2", Kind::F32, 1),
    param("cf_add_param1_coef3", Kind::F32, 1),
    param("cf_add_param2_coef1", Kind::F32, 1),
    param("cf_add_param2_coef2", Kind::F32, 1),
    param("cf_add_param2_coef3", Kind::F32, 1),
    // HP
    param("hp_speed_gap_param1", Kind::F32, 1),
    param("hp_speed_gap_param1_pow", Kind::F32, 1),
    param("hp_dec_base", Kind::F32, 1),
    param("hp_guts_coef", Kind::F32, 1),
    param("hp_guts_coef_sqrt", Kind::F32, 1),
    param("hp_rate_normal", Kind::F32, 1),
    param("hp_rate_tempt_ct_oonige", Kind::F32, 1),
    param("hp_rate_tempt_ct_nige", Kind::F32, 1),
    param("hp_rate_ct_oonige", Kind::F32, 1),
    param("hp_rate_ct_nige", Kind::F32, 1),
    param("hp_rate_temptation", Kind::F32, 1),
    param("hp_rate_pace_down", Kind::F32, 1),
    param("hp_rate_down_slope_multiply", Kind::F32, 1),
    // Last spurt
    param("ls_base_target_speed_add_coef", Kind::F32, 1),
    param("ls_distance_goal_buffer", Kind::I32, 1),
    param("ls_speed_delta", Kind::F32, 1),
    param("ls_step2_speed_delta", Kind::F32, 1),
    param("ls_step2_per_val1", Kind::F32, 1),
    param("ls_step2_per_val2", Kind::F32, 1),
    param("ls_guts_coef1", Kind::F32, 1),
    param("ls_guts_pow", Kind::F32, 1),
    param("ls_guts_coef2", Kind::F32, 1),
    param("ls_coef_sqrt", Kind::F32, 1),
    param("add_speed_param_coef", Kind::F32, 1),
    param("base_target_speed_coef", Kind::F32, 1),
    param("ls_distance_max_from_goal", Kind::F32, 1),
    // Lanes and overtaking
    param("force_in_move_lane_threshold", Kind::F32, 1),
    param("lane_move_speed_base", Kind::F32, 1),
    param("lane_move_speed_add", Kind::F32, 1),
    param("lane_move_speed_pow_coef", Kind::F32, 1),
    param("lane_move_speed_out_coef", Kind::F32, 1),
    param("lane_move_accel_coef", Kind::F32, 1),
    param("last_move_out_lane_normalize", Kind::F32, 1),
    param("last_move_out_lane_coef", Kind::F32, 1),
    param("last_move_out_lane_add_random", Kind::F32, 1),
    param("overtake_out_lane_coef_end", Kind::F32, 1),
    param("overtake_in_lane_coef_end", Kind::F32, 1),
    param("overtake_cool_down_time", Kind::F32, 1),
    param("overtake_dist_per_speed", Kind::F32, 1),
    // Position keep
    param("pk_cool_down_sec", Kind::I32, 1),
    param("pk_check_interval_sec", Kind::I32, 1),
    param("pk_pace_base_distance", Kind::I32, 1),
    param("pk_pace_base_distance_coef", Kind::F32, 1),
    // PositionKeepPaceUpDownDistanceDiff{Min,Max}{Senko,Sashi,Oikomi}, indexed by style (0, 1 unused)
    param("pk_diff_min", Kind::F32, 5),
    param("pk_diff_max", Kind::F32, 5),
    param("pk_diff_max_coef", Kind::F32, 1),
    param("pk_pace_down_target_lane", Kind::F32, 1),
    param("pk_pace_down_mult", Kind::F32, 1),
    param("pk_pace_down_mult_middle", Kind::F32, 1),
    param("pk_speed_up_per_val1", Kind::F32, 1),
    param("pk_speed_up_per_val1_oonige", Kind::F32, 1),
    param("pk_overtake_per_val1", Kind::F32, 1),
    param("pk_overtake_per_val1_oonige", Kind::F32, 1),
    param("pk_pace_up_per_val1", Kind::F32, 1),
    param("pk_speed_up_end_diff", Kind::F32, 1),
    param("pk_speed_up_end_diff_oonige", Kind::F32, 1),
    param("pk_speed_up_end_diff_one", Kind::F32, 1),
    param("pk_overtake_end_diff", Kind::F32, 1),
    param("pk_overtake_end_diff_oonige", Kind::F32, 1),
    // Temptation
    param("tm_end_check_time", Kind::F32, 1),
    param("tm_force_end_time", Kind::F32, 1),
    param("tm_end_per", Kind::F32, 1),
    param("tm_senko_to_nige", Kind::F32, 1),
    param("tm_sashi_to_senko", Kind::F32, 1),
    param("tm_sashi_to_nige", Kind::F32, 1),
    param("tm_oikomi_to_sashi", Kind::F32, 1),
    param("tm_oikomi_to_senko", Kind::F32, 1),
    param("tm_oikomi_to_nige", Kind::F32, 1),
    // Conserve power: the lookups of conserve_power.rs coef_by (1 when no row matches), by mode, style and
    // distance type
    param("cp_decrease_coef_temptation", Kind::F32, 1),
    param("cp_decrease_accel_coef_temptation", Kind::F32, 1),
    param("cp_increase_coef_pace_down", Kind::F32, 1),
    param("cp_increase_coef_normal", Kind::F32, 1),
    param("cp_style_coef", Kind::F32, 5),
    param("cp_release_time_distance_coef", Kind::F32, 5),
    param("cp_release_accel_coef_by", Kind::F32, 25),
    param("cp_release_time_coef", Kind::F32, 1),
    param("cp_releasable_time_threshold", Kind::F32, 1),
    param("cp_release_decl_coef", Kind::F32, 1),
    param("cp_release_accel_coef", Kind::F32, 1),
    param("cp_default_cool_down", Kind::F32, 1),
    param("finish_time_coef", Kind::F32, 1),
    // Skills
    param("lot_activate_per_val1", Kind::F32, 1),
    param("lot_activate_per_val2", Kind::F32, 1),
    param("activate_per_min", Kind::I32, 1),
];
