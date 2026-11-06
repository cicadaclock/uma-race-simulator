// physics/last_spurt.rs: where to start the last spurt and at what speed, from the HP left. The candidate list of
// the second pass lives in sort_key / ls_c_* (sort.glsl with SORT_CAP >= LS_MAX_CANDIDATES). Needs exact.glsl,
// race_state.glsl, net_random.glsl, sort.glsl, modifiers.glsl, queries.glsl, stats.glsl, transcend.glsl and
// speed.glsl.

#ifndef UMA_LAST_SPURT_GLSL
#define UMA_LAST_SPURT_GLSL

// speed_down_next pushes at most 101 candidates.
const int LS_MAX_CANDIDATES = 101;

// LastSpurtCalcParam of the calculation in progress.
float lsp_cur_distance, lsp_hp, lsp_speed, lsp_guts, lsp_wiz, lsp_bts, lsp_multi_hp_sub, lsp_pdc, lsp_rbs;
int lsp_phase, lsp_course_distance;
float ls_base_add;
int ls_goal_buffer;
float ls_c_distance[SORT_CAP], ls_c_speed[SORT_CAP];

// Distance left to the goal minus the goal buffer, at least 0.
float ls_remain_distance() {
    float r = (float(lsp_course_distance) - lsp_cur_distance) - float(ls_goal_buffer);
    return r > 0.0 ? r : 0.0;
}

float ls_hp_per_sec(float v) { return hp_per_sec(lsp_rbs, p_hp_rate_normal(), lsp_multi_hp_sub, lsp_phase, lsp_guts, v); }
float ls_need_hp(float v, float distance) { return calc_need_hp(lsp_rbs, lsp_multi_hp_sub, lsp_phase, lsp_guts, v, distance); }

// Longest spurt at vs (the rest at vn) the HP allows, searched in 30 m steps and starting before start_max.
bool ls_calc_spurt_distance(float vs, float vn, float start_max, out float start, out float time) {
    float remain = ls_remain_distance();
    float hs = ls_hp_per_sec(vs), hn = ls_hp_per_sec(vn);
    float d = remain;
    while (d > 0.0) {
        float t1 = div_rn32(d, vs);
        float t2 = div_rn32(remain - d, vn);
        // The comparison sees the f64 sum.
        if (double(lsp_hp) >= double(hs) * double(t1) + double(t2) * double(hn)) {
            float s = float(lsp_course_distance - ls_goal_buffer) - d;
            if (start_max > s) {
                start = s;
                time = t2 + t1;
                return true;
            }
        }
        d = d + -30.0;
    }
    start = 0.0;
    time = 0.0;
    return false;
}

// Lowers the speed from v until some spurt distance fits (at most 100 steps). Returns the result; *out_v the speed.
int ls_speed_down(float v, float start_max, out float out_v) {
    out_v = lsp_bts;
    float remain = ls_remain_distance();
    if (ls_need_hp(lsp_bts, remain) > lsp_hp) return LS_FALSE_BELOW_NEED_MIN_HP;
    if (!(lsp_hp < ls_need_hp(v, remain))) return LS_TRUE_EXCEED_NEED_MAX_HP;
    float lo = lsp_bts > v ? v : lsp_bts;
    if (v < lo) return LS_FALSE;
    int k = 0;
    while (true) {
        float d, t;
        if (ls_calc_spurt_distance(v, lsp_bts, start_max, d, t)) {
            out_v = v;
            return LS_TRUE;
        }
        v = v - p_ls_speed_delta();
        if (lo >= v) return LS_FALSE;
        k++;
        if (k > 100 || !(v >= lo)) return LS_FALSE;
    }
    return LS_FALSE;
}

// Second pass from v down: every feasible speed, sorted by need time with the truncating comparator, then one
// Ri(100) per candidate tried until one passes the Wit check (else the last candidate).
void ls_speed_down_next(float v, float start_max, out float out_d, out float out_v) {
    float v_in = v;
    float lo = lsp_bts > v ? v : lsp_bts;
    int n = 0, k = 0;
    while (v >= lo) {
        float d, t;
        if (ls_calc_spurt_distance(v, lsp_bts, start_max, d, t)) {
            if (n >= SORT_CAP) {
                gpu_fault(FAULT_LAST_SPURT_SORT);
            } else {
                ls_c_distance[n] = d;
                ls_c_speed[n] = v;
                sort_key[n] = t;
                n++;
            }
        }
        v = v - p_ls_step2_speed_delta();
        if (lo >= v) break;
        k++;
        if (k > 100) break;
    }
    if (n == 0) {
        out_d = lsp_cur_distance;
        out_v = v_in;
        return;
    }
    for (int j = 0; j < n; j++) sort_h[j] = j;
    sort_mode = SORT_KEY_TRUNC;
    if (!intro_sort(n)) gpu_fault(FAULT_LAST_SPURT_SORT);
    float per = float(double(lsp_wiz) * double(p_ls_step2_per_val2()) + double(p_ls_step2_per_val1()));
    int chosen = -1;
    for (int j = 0; j < n; j++) {
        int r = race_random_int(0u, 100);
        if (per > float(r)) {
            chosen = sort_h[j];
            break;
        }
    }
    if (chosen < 0) chosen = sort_h[n - 1];
    out_d = ls_c_distance[chosen];
    out_v = ls_c_speed[chosen];
}

// calc_last_spurt on a fresh LastSpurt (start -1, target speed = race base speed, result False).
void calc_last_spurt(out int result, out float start_distance, out float target_speed) {
    start_distance = -1.0;
    target_speed = lsp_rbs;
    // Max spurt speed g + (s + b), the whole sum in f64 with one rounding.
    double g = double(mathf_pow(lsp_guts * p_ls_guts_coef1(), p_ls_guts_pow())) * double(p_ls_guts_coef2());
    double s = double(sqrt_rn32(p_ls_coef_sqrt() * lsp_speed)) * double(lsp_pdc) * double(p_add_speed_param_coef());
    double b = double(ls_base_add + lsp_bts) * double(p_base_target_speed_coef());
    float max_v = float(g + (s + b));
    float start_max = float(lsp_course_distance) - p_ls_distance_max_from_goal();
    float out_v;
    int r = ls_speed_down(max_v, start_max, out_v);
    result = r;
    if ((r & LS_TRUE_FLAGS) == 0) {
        start_distance = -1.0;
    } else {
        if (r == LS_TRUE) {
            float d2, v2;
            ls_speed_down_next(out_v, start_max, d2, v2);
            start_distance = d2;
            target_speed = v2;
        } else if (r == LS_TRUE_EXCEED_NEED_MAX_HP) {
            start_distance = lsp_cur_distance;
            target_speed = max_v;
        }
        if (lsp_cur_distance > start_distance) start_distance = lsp_cur_distance;
    }
}

// Recomputes the last spurt when requested and stores the result on the AI.
void update_last_spurt(uint i) {
    if (!h_req_calc_ls(i)) return;
    lsp_cur_distance = h_distance(i);
    lsp_hp = h_hp(i);
    lsp_phase = h_phase(i);
    lsp_speed = stat_speed(i);
    lsp_guts = stat_guts(i);
    lsp_wiz = stat_wiz_for(i, WIZ_LAST_SPURT);
    lsp_bts = base_target_speed_without_random(i, h_phase(i));
    lsp_multi_hp_sub = h_multi_hp_sub(i);
    lsp_pdc = h_proper_distance_coef_speed(i);
    lsp_course_distance = r_course_distance();
    lsp_rbs = r_base_speed();
    ls_base_add = r_base_speed() * p_ls_base_target_speed_add_coef();
    ls_goal_buffer = p_ls_distance_goal_buffer();
    int result;
    float start, speed;
    calc_last_spurt(result, start, speed);
    int count = h_ls_calc_count(i) + 1;
    set_h_ls_calc_count(i, count);
    if (count <= 1) {
        set_h_ls_calc_result(i, result);
        set_h_ls_target_speed(i, speed);
        set_h_ls_start_distance(i, start);
    } else if ((result & LS_TRUE_FLAGS) != 0) {
        // A recompute that yields False keeps the old values; an existing start point is kept.
        set_h_ls_calc_result(i, result);
        set_h_ls_target_speed(i, speed);
        if (0.0 > h_ls_start_distance(i)) set_h_ls_start_distance(i, start);
    }
    set_h_req_calc_ls(i, false);
}

#endif
