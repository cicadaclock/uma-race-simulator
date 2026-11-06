// ai/lane.rs: the race lane planner and its strategies (Normal / Overtake / Fixed), force-in, and the lane move.
// The overtake candidate list is a local array here (the Rust keeps it as scratch on the planner; nothing reads it
// between frames). Needs exact.glsl, race_state.glsl, net_random.glsl, modifiers.glsl and queries.glsl.

#ifndef UMA_LANE_GLSL
#define UMA_LANE_GLSL

#define CROWD_LANE_OFFSET uintBitsToFloat(0x3d638e3au)  // one ulp above 1/18
#define HORSE_LANE uintBitsToFloat(0x3d638e39u)  // 1/18
#define BOX_HALF uintBitsToFloat(0x3d360b62u)  // 0.8 horse lanes
#define LANE_STEP uintBitsToFloat(0x3d4ccccdu)  // 0.05
#define LANE_1_36 uintBitsToFloat(0x3ce38e39u)  // 0.027777778
#define LANE_1_72 uintBitsToFloat(0x3c638e39u)  // 0.013888889
#define LANE_1_9 uintBitsToFloat(0x3de38e39u)  // 0.111111112
#define FLT_MAX uintBitsToFloat(0x7f7fffffu)

float clamp_lane(uint h, float v) { return mathf_clamp(v, h_lane_distance_min(h), h_lane_distance_max(h)); }

bool is_target_lane_fixed(uint h) { return !mathf_approximately(apply_modifier(h, MOD_TARGET_LANE, 0.0), 0.0); }

// The lane of the last move out: one GetRandom(0f, addRandom).
float calc_move_out_lane(uint i) {
    float nv = p_last_move_out_lane_normalize();
    float norm = mathf_approximately(nv, 0.0) ? uintBitsToFloat(0x3e4ccccdu) : nv;  // 0.2f
    float r = mathf_clamp01(div_rn32(h_lane_distance(i), norm));
    float rnd = race_random_float_range(0u, 0.0, p_last_move_out_lane_add_random());
    return mathf_clamp(float(double(r) * double(p_last_move_out_lane_coef()) + double(rnd)), h_lane_distance_min(i), h_lane_distance_max(i));
}

float add_lane_by_direction(float lane, bool is_in, float t) {
    return is_in ? mathf_max(t, lane - LANE_STEP) : mathf_min(t, lane + LANE_STEP);
}

float search_normal(uint i) {
    float lane = h_lane_distance(i);
    if (0.0 >= h_hp(i)) return lane;
    if (pk_mode(i) == PK_PACE_DOWN) return p_pk_pace_down_target_lane();
    if (h_lp_use_extra(i)) {
        if (h_lp_extra_lane(i) <= lane || !is_enable_out_move(i)) return lane;
        return add_lane_by_direction(lane, false, h_lp_extra_lane(i));
    }
    if (h_phase(i) > PHASE_MIDDLE) return lane;
    if (is_enable_in_move(i)) return lane - LANE_STEP;
    if (h_phase(i) != PHASE_MIDDLE) return lane;
    int b = h_block_horse_in(i);
    if (b < 0) return lane;
    if (!(p_side_block_lane_gap() - LANE_1_72 > h_ar_lane_gap_abs(i, uint(b)))) return lane;
    return add_lane_by_direction(lane, false, h_lane_distance(uint(b)) + p_side_block_lane_gap());
}

float search_fixed(uint i) {
    float v = apply_modifier(i, MOD_TARGET_LANE, 0.0);
    float lane = h_lane_distance(i);
    if (lane > v) {
        if (is_enable_in_move(i)) return add_lane_by_direction(lane, true, v);
    } else if (v > lane && is_enable_out_move(i)) {
        return add_lane_by_direction(lane, false, v);
    }
    return lane;
}

float lane_gap_coef_by_phase(uint i, float gap) {
    int ph = h_phase(i);
    if (ph == PHASE_START) return gap > 0.0 ? 100.0 : 1.0;
    if (ph == PHASE_MIDDLE) return 1.0;
    return gap > 0.0 ? p_overtake_out_lane_coef_end() : p_overtake_in_lane_coef_end();
}

float lane_gap_coef_by_skill(uint i, float gap) {
    bool f_in = has_modifier(i, MOD_FORCE_OVERTAKE_IN), f_out = has_modifier(i, MOD_FORCE_OVERTAKE_OUT);
    return (f_in != f_out && ((f_in && gap > 0.0) || (f_out && 0.0 > gap))) ? 1.5 : 1.0;
}

bool check_enable_move_lane(uint i, float c) {
    if (0.0 > c || c > r_info_lane_distance_max()) return false;
    float d = c - h_lane_distance(i);
    if (0.0 > d) return is_enable_in_move(i) && check_lane_space(i, d);
    if (d > 0.0) return is_enable_out_move(i) && check_lane_space(i, d);
    return true;
}

// Any visible horse of i (live distance and lane) inside the box.
bool check_horse_exist(uint i, float d_min, float d_max, float l_min, float l_max) {
    for (int k = 0; k < h_visible_horse_count(i); k++) {
        uint o = uint(h_visible_horses(i, uint(k)));
        float d = h_distance(o), l = h_lane_distance(o);
        if (!(d_min > d) && !(d > d_max) && !(l_min > l) && !(l > l_max)) return true;
    }
    return false;
}

// The horse at the inner (outer) edge of the crowd around `start`: repeatedly the furthest-in (out) horse within
// 3 m ahead and g lanes. Gap filters use each horse's own scan; lane comparisons are live. (Tail recursion in the
// Rust; every step moves strictly in or out, so it ends within the field.)
uint crowd_edge_horse(uint start, float g, bool is_in) {
    uint cur = start;
    for (int step = 0; step <= int(MAX_HORSES); step++) {
        float best_lane = h_lane_distance(cur);
        int best = -1;
        uint left = h_ar_others(cur);
        while (left != 0u) {
            uint j = uint(findLSB(left));
            left &= left - 1u;
            float distance_gap = h_ar_distance_gap(cur, j), lane_gap = h_ar_lane_gap(cur, j);
            if (0.0 > distance_gap || distance_gap > 3.0 || h_ar_lane_gap_abs(cur, j) >= g) continue;
            if (is_in ? !(0.0 > lane_gap) : !(lane_gap > 0.0)) continue;
            float l = h_lane_distance(j);
            if ((is_in && l < best_lane) || (!is_in && l > best_lane)) {
                best_lane = l;
                best = int(j);
            }
        }
        if (best < 0) return cur;
        cur = uint(best);
    }
    gpu_fault(FAULT_UPDATE_HORSES);
    return cur;
}

// Overtaking: the best free lane beside the targets (or straight ahead), weighted by phase and skills.
float search_overtake(uint i) {
    float lane = h_lane_distance(i);
    if (!is_overtake_or_cool_down(i)) return lane;
    float cand[2 * MAX_HORSES + 1u];
    int nc = 0;
    float min_target_dist = FLT_MAX;
    float d0 = h_distance(i);
    for (int t = 0; t < h_ot_count(i); t++) {
        uint o = uint(h_ot_targets(i, uint(t)));
        if (min_target_dist > h_distance(o)) min_target_dist = h_distance(o);
        uint in_edge = crowd_edge_horse(o, LANE_1_9, true);
        float in_lane = h_lane_distance(in_edge) - CROWD_LANE_OFFSET;
        uint out_edge = crowd_edge_horse(o, LANE_1_9, false);
        float out_lane = h_lane_distance(out_edge) + CROWD_LANE_OFFSET;
        if (in_lane >= h_lane_distance_min(i) && !check_horse_exist(i, d0, h_distance(in_edge) + 0.5, in_lane - BOX_HALF, in_lane + BOX_HALF)) {
            cand[nc++] = in_lane;
        }
        if (h_lane_distance_max(i) >= out_lane && !check_horse_exist(i, d0, h_distance(out_edge) + 0.5, out_lane - BOX_HALF, out_lane + BOX_HALF)) {
            cand[nc++] = out_lane;
        }
    }
    if (h_phase(i) <= PHASE_MIDDLE && is_enable_in_move(i) && lane > HORSE_LANE) {
        if (!check_horse_exist(i, d0, min_target_dist + 3.0, (lane - BOX_HALF) - HORSE_LANE, (lane + BOX_HALF) - HORSE_LANE)) cand[nc++] = lane - HORSE_LANE;
    } else if (!check_horse_exist(i, d0, min_target_dist + 3.0, lane - BOX_HALF, lane + BOX_HALF)) {
        cand[nc++] = lane;
    }
    // Lowest |score| wins, first wins ties.
    float best = lane, best_score = FLT_MAX;
    for (int k = 0; k < nc; k++) {
        float c = cand[k];
        float gap = lane - c;
        float score = abs(float((double(lane_gap_coef_by_phase(i, gap)) * double(gap)) * double(lane_gap_coef_by_skill(i, gap))));
        if (score < best_score && check_enable_move_lane(i, c)) {
            best = c;
            best_score = score;
        }
    }
    // A further-out last-move-out or skill-pinned lane overrides a pick that is not inward.
    if (h_lp_use_extra(i)) {
        float x = h_lp_extra_lane(i);
        if (best < lane || x <= best || !check_enable_move_lane(i, x)) return best;
        return x;
    }
    if (is_target_lane_fixed(i)) {
        if (best < lane) return best;
        float v = apply_modifier(i, MOD_TARGET_LANE, 0.0);
        if (v <= best || !check_enable_move_lane(i, v)) return best;
        return v;
    }
    return best;
}

// Keeps the current target while still moving toward it, unless a re-check is forced or the horse is overtaking.
// At most one RNG draw (the last move out).
void update_race_lane_planner(uint i) {
    if (h_lp_force_check(i)) {
        set_h_lp_force_check(i, false);
    } else if (!is_overtake(i)) {
        float d = h_lp_target_lane(i) - h_lane_distance(i);
        if (!(LANE_1_36 > d && d > -LANE_1_36)) {
            bool enabled = d > 0.0 ? is_enable_out_move(i) : is_enable_in_move(i);
            if (enabled) return;
        }
    }
    set_h_lp_target_lane(i, clamp_lane(i, h_lane_distance(i)));
    float lmo = h_lp_last_move_out_start(i);
    if (!h_lp_extra_checked(i) && !(0.0 > lmo) && h_distance(i) >= lmo) {
        set_h_lp_extra_lane(i, calc_move_out_lane(i));
        set_h_lp_extra_checked(i, true);
        set_h_lp_use_extra(i, true);
    }
    float fmlp = r_first_move_lane_point();
    bool overtake = is_overtake_or_cool_down(i) && (h_phase(i) > PHASE_MIDDLE || fmlp < h_distance(i) || !(h_distance(i) >= fmlp - 200.0));
    float v;
    if (overtake) v = search_overtake(i);
    else if (is_target_lane_fixed(i)) v = search_fixed(i);
    else v = search_normal(i);
    set_h_lp_target_lane(i, clamp_lane(i, v));
}

// The race planner when active; a Fixed planner leaves the target lane unchanged.
void update_move_lane(uint i) {
    if (h_active_lane_planner(i) == 0 && h_lp_is_race(i)) {
        update_race_lane_planner(i);
        set_h_target_lane(i, h_lp_target_lane(i));
    }
}

// Force-in (a target-speed boost): before the middle phase, at or outside the threshold lane and unable to move in.
void update_force_in_move(uint i) {
    bool v = h_is_empty_in_move(i) && h_phase(i) < PHASE_MIDDLE && !(p_force_in_move_lane_threshold() > h_lane_distance(i)) && !is_enable_in_move(i);
    set_h_force_in_move(i, v);
}

// Lane-change speed magnitude: accelerates toward a power-based target, then skills apply (clamped to [0, 0.6]).
float calc_lane_move_speed(uint i, float dt) {
    float base = p_lane_move_speed_base(), add = p_lane_move_speed_add();
    float pw = mathf_clamp(apply_modifier(i, MOD_POWER, h_base_pow_adjusted(i)), 1.0, 2000.0);
    float target = float((double(pw) * double(p_lane_move_speed_pow_coef()) + double(add)) * double(base));
    if (h_phase(i) < PHASE_END) {
        if (r_first_move_lane_point() >= h_distance(i)) {
            float rate = div_rn32(h_lane_distance(i), h_lane_distance_max(i));
            target = float(double(target) * (double(rate) * double(p_lane_move_speed_out_coef()) + 1.0lf));
        }
    } else {
        target = float(double(target) * (double(h_cur_order(i)) * double(uintBitsToFloat(0x3c23d70au)) + 1.0lf));  // 0.01f
    }
    float accel = p_lane_move_accel_coef() * p_lane_move_speed_base();
    float cur = h_last_lane_move_speed(i);
    if (target > cur) {
        cur = mathf_min(target, float(double(accel) * double(dt) + double(cur)));
    } else if (cur > target) {
        cur = float(double(cur) - double(accel) * double(dt));
        if (target > cur) cur = target;
    }
    cur = mathf_max(cur, 0.0);
    set_h_last_lane_move_speed(i, cur);
    return mathf_clamp(apply_modifier(i, MOD_LANE_MOVE_SPEED, cur), 0.0, uintBitsToFloat(0x3f19999au));  // 0.6f
}

void move_lane(uint i, float dt) {
    float prev = h_lane_distance(i);
    set_h_prev_lane_distance(i, prev);
    float target = h_target_lane(i);
    if (mathf_approximately(prev, target)) {
        set_h_lane_distance(i, mathf_clamp(prev, h_lane_distance_min(i), h_lane_distance_max(i)));
        set_h_move_lane_continue_time(i, 0.0);
        return;
    }
    int old_dir = h_move_lane_dir(i);
    int dir = prev <= target ? 1 : 0;
    set_h_move_lane_dir(i, dir);
    float speed;
    if (h_is_finished(i) && h_after_goal_time(i) >= 1.0) {
        // After the goal the lane speed is literally dt.
        speed = prev > target ? dt * -1.0 : dt;
    } else {
        float s = calc_lane_move_speed(i, dt);
        speed = prev > target ? float((double(prev) + 1.0lf) * -double(s)) : s;
    }
    set_h_lane_move_speed(i, speed);
    if (mathf_approximately(speed, 0.0)) return;
    float nl = float(double(dt) * double(speed) + double(prev));
    if (prev > target) nl = mathf_max(target, nl);
    else if (nl > target) nl = target;
    nl = mathf_clamp(nl, h_lane_distance_min(i), h_lane_distance_max(i));
    if (nl > prev) {
        if (!is_enable_out_move(i)) return;
    } else if (prev > nl && !is_enable_in_move(i)) {
        return;
    }
    set_h_lane_distance(i, nl);
    if (old_dir == dir) set_h_move_lane_continue_time(i, dt + h_move_lane_continue_time(i));
    if (mathf_approximately(nl, target)) set_h_last_lane_move_speed(i, 0.0);
}

#endif
