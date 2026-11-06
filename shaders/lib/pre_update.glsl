// Stage::PreUpdate: the scan-view snapshot, then per horse tick.rs pre_update: the around table
// (ai/around.rs update_around_horses_param), the after-goal timer, the modifier timers and the corner reset.
// Needs exact.glsl, race_state.glsl and modifiers.glsl.

#ifndef UMA_PRE_UPDATE_GLSL
#define UMA_PRE_UPDATE_GLSL

bool is_straight_last(uint h) { return h_is_straight(h) && h_is_final_corner(h); }

// Vision depth after skills, at least 0 (NaN -> 0).
float visible_distance(uint h) { return mathf_max(apply_modifier(h, MOD_VISIBLE_DISTANCE, h_base_visible_distance(h)), 0.0); }

// |x|, but -0 stays -0 (as in the game).
float abs_keep_neg_zero(float x) { return 0.0 > x ? -x : x; }

bool check_front_block(float distance_gap, float lane_gap_abs) {
    float fbdg = p_front_block_distance_gap();
    if (0.0 >= distance_gap || distance_gap >= fbdg) return false;
    float tol = float((1.0lf - div_rn64(double(distance_gap), double(fbdg)) * double(uintBitsToFloat(0x3F19999Au))) * double(p_front_block_lane_gap()));
    return !(lane_gap_abs > tol);
}

bool is_visible(float v, float distance_gap, float lane_gap_abs) {
    if (0.0 >= v || 0.0 > distance_gap || distance_gap > v) return false;
    double w = (div_rn64(double(distance_gap), double(v)) * double(uintBitsToFloat(0x3F238E39u)) + double(uintBitsToFloat(0x3DE38E39u))) * 0.5lf;
    return !(lane_gap_abs > float(w));
}

void snapshot_scan_view() {
    uint mask = 0u;
    for (uint j = 0u; j < uint(r_num_horses()); j++) {
        set_r_sv_distance(j, h_distance(j));
        set_r_sv_lane(j, h_lane_distance(j));
        set_r_sv_last_speed(j, h_last_speed(j));
        if (is_straight_last(j)) mask |= 1u << j;
    }
    set_r_sv_straight_last(mask);
}

uint lowest(uint set) { return uint(findLSB(set)); }

void update_around_horses_param(uint i, float dt) {
    uint n = uint(r_num_horses());
    float vision = visible_distance(i);
    float my_distance = r_sv_distance(i), my_lane_sv = r_sv_lane(i), my_speed = r_sv_last_speed(i);
    float ssd = p_surround_side_distance(), ssl = p_surround_side_lane(), sd = p_surround_distance(), sl = p_surround_lane();
    float nd = p_near_distance(), nl = p_near_lane(), cfl = p_compete_fight_lane_gap(), cfd = p_compete_fight_distance_gap();
    float sbd = p_side_block_distance_gap(), sbl = p_side_block_lane_gap(), fbd = p_front_block_distance_gap();
    uint surround_out = 0u, surround_ahead = 0u, surround_behind = 0u, near = 0u, cf_lane = 0u, cf_far = 0u;
    uint back = 0u, front = 0u, side = 0u, lane_outside = 0u, lane_inside = 0u, side_near = 0u, seen = 0u;
    for (uint j = 0u; j < n; j++) {
        float distance = r_sv_distance(j) - my_distance;
        float distance_abs = abs_keep_neg_zero(distance);
        float lane = r_sv_lane(j) - my_lane_sv;
        float lane_abs = abs_keep_neg_zero(lane);
        float speed = my_speed - r_sv_last_speed(j);
        set_h_ar_distance_gap(i, j, distance);
        set_h_ar_distance_gap_abs(i, j, distance_abs);
        set_h_ar_lane_gap(i, j, lane);
        set_h_ar_lane_gap_abs(i, j, lane_abs);
        set_h_ar_speed_gap(i, j, speed);
        set_h_ar_speed_gap_abs(i, j, abs_keep_neg_zero(speed));
        uint bit = 1u << j;
        if (ssd >= distance_abs && lane >= 0.0 && ssl >= lane) surround_out |= bit;
        if (distance > 0.0 && sd >= distance && sl >= lane_abs) surround_ahead |= bit;
        if (0.0 > distance && distance >= -sd && sl >= lane_abs) surround_behind |= bit;
        if (!(distance_abs > nd) && !(lane_abs > nl)) near |= bit;
        if (cfl >= lane_abs) cf_lane |= bit;
        if (cfd < distance_abs) cf_far |= bit;
        if (-sbd > distance) back |= bit;
        if (distance > 0.0 && distance < fbd) front |= bit;
        if (distance_abs < sbd && lane_abs < sbl && lane_abs != 0.0) side |= bit;
        if (lane > 0.0) lane_outside |= bit;
        if (0.0 > lane) lane_inside |= bit;
        if (sbd > distance_abs) side_near |= bit;
        if (0.0 <= distance && distance <= vision) seen |= bit;
    }
    uint others = h_ar_others(i);
    set_h_ar_back(i, back & others);
    // The near / visible lists are not cleared: only their first `count` entries are ever read.
    uint near_set = near & others;
    int near_count = bitCount(near_set);
    uint at = 0u;
    while (near_set != 0u) {
        set_h_near_horses(i, at, int(lowest(near_set)));
        at++;
        near_set &= near_set - 1u;
    }
    // Compete-fight candidates: close, side by side, and either horse on the final straight.
    uint straight = is_straight_last(i) ? others : (r_sv_straight_last() & others);
    uint candidates = cf_lane & ~cf_far & straight & others;
    uint leaving = others & ~candidates;
    while (leaving != 0u) {
        uint j = lowest(leaving);
        int len = h_cf_near_len(i);
        for (int k = 0; k < len; k++) {
            if (h_cf_near(i, uint(k)) == int(j)) {
                for (int m = k; m + 1 < len; m++) set_h_cf_near(i, uint(m), h_cf_near(i, uint(m + 1)));
                set_h_cf_near(i, uint(len - 1), 0);
                set_h_cf_near_len(i, len - 1);
                break;
            }
        }
        set_h_ar_cf_time(i, j, 0.0);
        leaving &= leaving - 1u;
    }
    uint entering = candidates;
    while (entering != 0u) {
        uint j = lowest(entering);
        int len = h_cf_near_len(i);
        bool found = false;
        for (int k = 0; k < len; k++) found = found || h_cf_near(i, uint(k)) == int(j);
        if (found) set_h_ar_cf_time(i, j, dt + h_ar_cf_time(i, j));
        else {
            set_h_cf_near(i, uint(len), int(j));
            set_h_cf_near_len(i, len + 1);
        }
        entering &= entering - 1u;
    }
    uint ahead = others & ~back;
    float min_front = uintBitsToFloat(0x7f7fffffu);
    int block_horse = -1;
    uint fr = front & ahead;
    while (fr != 0u) {
        uint j = lowest(fr);
        float distance = h_ar_distance_gap(i, j);
        if (check_front_block(distance, h_ar_lane_gap_abs(i, j)) && min_front > distance) {
            min_front = distance;
            block_horse = int(j);
        }
        fr &= fr - 1u;
    }
    uint sd_set = side & ahead;
    set_h_block_horse_out(i, findMSB(sd_set & lane_outside));
    set_h_block_horse_in(i, findMSB(sd_set & ~lane_outside));
    set_h_block_horse(i, block_horse);
    float my_lane = h_lane_distance(i);
    float in_lane_space = my_lane;
    uint near_inside = side_near & ahead & lane_inside;
    while (near_inside != 0u) {
        float gap = h_ar_lane_gap_abs(i, lowest(near_inside));
        if (in_lane_space > gap) in_lane_space = gap;
        near_inside &= near_inside - 1u;
    }
    float out_lane_space = h_lane_distance_max(i) - my_lane;
    uint near_outside = side_near & ahead & lane_outside;
    while (near_outside != 0u) {
        float gap = h_ar_lane_gap_abs(i, lowest(near_outside));
        if (out_lane_space > gap) out_lane_space = gap;
        near_outside &= near_outside - 1u;
    }
    int visible_count = 0;
    if (vision > 0.0) {
        uint s = seen & ahead;
        while (s != 0u) {
            uint j = lowest(s);
            if (is_visible(vision, h_ar_distance_gap(i, j), h_ar_lane_gap_abs(i, j))) {
                set_h_visible_horses(i, uint(visible_count), int(j));
                visible_count++;
            }
            s &= s - 1u;
        }
    }
    set_h_is_around_horse_initialized(i, true);
    set_h_in_lane_space(i, in_lane_space);
    set_h_out_lane_space(i, out_lane_space);
    set_h_near_horse_count(i, near_count);
    set_h_visible_horse_count(i, visible_count);
    set_h_is_surround_out(i, (surround_out & others) != 0u);
    set_h_is_surround_ahead(i, (surround_ahead & others) != 0u);
    set_h_is_surround_behind(i, (surround_behind & others) != 0u);
}

void update_after_goal_time(uint i, float dt) {
    if (!h_is_finished(i)) return;
    float old = h_after_goal_time(i);
    set_h_after_goal_time(i, dt + old);
    if (1.0 > old && h_after_goal_time(i) >= 1.0) set_h_active_lane_planner(i, 1);
}

void pre_update(float dt) {
    snapshot_scan_view();
    for (uint i = 0u; i < uint(r_num_horses()); i++) {
        set_h_move_distance(i, 0.0);
        update_around_horses_param(i, dt);
        update_after_goal_time(i, dt);
        update_modifiers(i, dt);
        if (h_cur_corner(i) != 0 && h_corner_end_distance(i) < h_distance(i)) set_h_cur_corner(i, 0);
    }
}

#endif
