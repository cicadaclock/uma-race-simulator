// Enum values and small state queries shared by the stages. Needs exact.glsl and race_state.glsl.

#ifndef UMA_QUERIES_GLSL
#define UMA_QUERIES_GLSL

// defs.rs
const int STYLE_NONE = 0, STYLE_NIGE = 1, STYLE_SENKO = 2, STYLE_SASHI = 3, STYLE_OIKOMI = 4;
const int STYLE_EX_NONE = 0, STYLE_EX_OONIGE = 1;
const int PHASE_START = 0, PHASE_MIDDLE = 1, PHASE_END = 2, PHASE_LAST = 3, PHASE_FINISHED = 4;
const int SLOPE_NULL = 0, SLOPE_UP = 1, SLOPE_DOWN = 2;
const int PK_NULL = 0, PK_SPEED_UP = 1, PK_OVERTAKE = 2, PK_PACE_UP = 3, PK_PACE_DOWN = 4, PK_PACE_UP_EX = 5;
// last_spurt_calc_result
const int LS_TRUE_EXCEED_NEED_MAX_HP = 1, LS_TRUE = 2, LS_FALSE_BELOW_NEED_MIN_HP = 4, LS_FALSE = 8;
const int LS_TRUE_FLAGS = 3, LS_FALSE_FLAGS = 12;
// ai::temptation::TemptationMode
const int TM_NULL = 0, TM_POSITION_SASHI = 1, TM_POSITION_SENKO = 2, TM_POSITION_NIGE = 3, TM_BOOST = 4;

// gpu_fault bits. Elsewhere: 1, 2, 4, 16 a sort's heapsort fallback (order, finish, end of frame, field), 8 a full
// modifier pool (modifiers.glsl).
const uint FAULT_TRANSCEND = 32u, FAULT_UPDATE_HORSES = 64u, FAULT_LAST_SPURT_SORT = 128u;
void gpu_fault(uint bit) { set_r_gpu_fault(r_gpu_fault() | bit); }

// field::horse_index_at_order: 0 when out of range.
int horse_index_at_order(int order) {
    return (order >= 0 && order < r_num_horses()) ? r_order_horse(uint(order)) : 0;
}
// field::horse_at_order: -1 when out of range.
int horse_at_order(int order) {
    return (order >= 0 && order < r_num_horses()) ? r_order_horse(uint(order)) : -1;
}

// physics::last_spurt::is_last_spurt
bool is_last_spurt(uint h) {
    return h_ls_start_distance(h) > 0.0 && (h_ls_calc_result(h) & LS_FALSE_FLAGS) == 0 && h_distance(h) >= h_ls_start_distance(h);
}

// lane::set_force_check_next_lane_one_time: on the active lane planner.
void set_force_check_next_lane_one_time(uint h, bool v) {
    if (h_active_lane_planner(h) == 0) set_h_lp_force_check(h, v);
    else set_h_agl_force_check(h, v);
}

// around.rs: free to move in / out, room for a lane change.
bool is_enable_in_move(uint h) {
    return h_block_horse_in(h) < 0 && !(h_lane_distance_min(h) + uintBitsToFloat(0x38d1b717u) >= h_lane_distance(h));  // 0.0001f
}
bool is_enable_out_move(uint h) { return h_block_horse_out(h) < 0 && !(h_lane_distance(h) >= h_lane_distance_max(h)); }
bool check_lane_space(uint h, float lane_gap_range) {
    return 0.0 > lane_gap_range ? h_in_lane_space(h) >= -lane_gap_range : h_out_lane_space(h) >= lane_gap_range;
}

// position_keep.rs: the current mode (cur_strategy is the mode's index).
int pk_mode(uint h) { return h_pk_cur(h); }
bool is_position_keep(uint h) { return pk_mode(h) != PK_NULL; }

// overtake.rs
bool is_overtake(uint h) { return h_ot_count(h) > 0; }
bool is_overtake_or_cool_down(uint h) { return is_overtake(h) || h_ot_cool_down(h) > 0.0; }

#endif
