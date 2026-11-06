// ai/position_keep.rs: the early-race speed-up / overtake / pace modes. Needs exact.glsl, race_state.glsl,
// net_random.glsl, modifiers.glsl, queries.glsl, stats.glsl and transcend.glsl.

#ifndef UMA_POSITION_KEEP_GLSL
#define UMA_POSITION_KEEP_GLSL

float pk_base_target_speed_multiply(uint h) { return h_pk_mult(h, uint(h_pk_cur(h))); }

// Back to no mode, with a cooldown before the next start.
void pk_clear(uint h) {
    set_h_pk_cur(h, PK_NULL);
    set_h_pk_cool_down(h, float(p_pk_cool_down_sec()));
}

// field.pacemaker.current(): the pacemaker, set to horse 0 when there is none.
uint pk_pacemaker() {
    if (r_pm_horse() < 0) set_r_pm_horse(0);
    return uint(r_pm_horse());
}

// When the rush starts or ends: reset the timers and switch to the style the rush imposes (or back).
void pk_check_temptation(uint h) {
    bool t = h_is_temptation(h);
    if (t == h_pk_temptation_prev(h)) return;
    set_h_pk_temptation_prev(h, t);
    pk_clear(h);
    set_h_pk_cool_down(h, 0.0);
    set_h_pk_check_interval(h, 0.0);
    int style = h_running_style(h);
    if (t) {
        int m = h_tm_mode(h);
        style = m == TM_POSITION_SASHI ? STYLE_SASHI : (m == TM_POSITION_SENKO ? STYLE_SENKO : STYLE_NIGE);
    }
    set_h_pk_style(h, style);
    set_h_pk_force_speed_up(h, style == STYLE_NIGE && t);
}

// Whether any horse ahead of `cur_order` has a more backward running style than `style`.
bool pk_backward_style_ahead(int cur_order, int style) {
    for (int o = cur_order - 1; o >= 0; o--) {
        if (h_running_style(uint(r_order_horse(uint(o)))) > style) return true;
    }
    return false;
}

// The (min, max) distance band behind the pacemaker for a style, widened on long courses.
vec2 pk_offset_distance(int course_distance, int style) {
    float coef = float(double(float(course_distance - p_pk_pace_base_distance())) * double(p_pk_pace_base_distance_coef()) + 1.0lf);
    if (1.0 > coef) coef = 1.0;
    if (style == STYLE_SENKO) return vec2(p_pk_diff_min(2u), coef * p_pk_diff_max(2u));
    if (style == STYLE_SASHI || style == STYLE_OIKOMI) return vec2(coef * p_pk_diff_min(uint(style)), coef * p_pk_diff_max(uint(style)));
    return vec2(0.0);
}

int pk_calc_keep_mode(uint i) {
    if (h_pk_style(i) == STYLE_NIGE) {
        if (h_pk_force_speed_up(i)) return PK_SPEED_UP;
        if (!r_pre_2026_07() && pk_backward_style_ahead(h_cur_order(i), h_running_style(i))) return PK_PACE_UP_EX;
        if (h_cur_order(i) == 0 || r_style_top_horse(uint(STYLE_NIGE)) == int(i)) return PK_SPEED_UP;
        return PK_OVERTAKE;
    }
    uint pm = pk_pacemaker();
    if (!r_pre_2026_07() && h_cur_order(pm) < h_cur_order(i) && h_running_style(i) < h_running_style(pm)) return PK_PACE_UP_EX;
    vec2 band = pk_offset_distance(r_course_distance(), h_pk_style(i));
    float gap = h_distance(pm) - h_distance(i);
    if (gap < band.x) return pm != i ? PK_PACE_DOWN : PK_NULL;
    if (!(gap > band.y)) return PK_NULL;
    return PK_PACE_UP;
}

// One GetRandom(100f): success iff r < log10(Wiz * 0.1) * val1.
bool pk_wiz_roll(uint i, float val1) {
    float r = race_random_float(0u, 100.0);
    float per = mathf_log10(stat_wiz_for(i, WIZ_POSITION_KEEP) * uintBitsToFloat(0x3dcccccdu)) * val1;  // 0.1f
    return !(per <= r);
}

// How far ahead of the horse behind a SpeedUp ends.
float pk_speed_up_end_distance_diff(uint i) {
    if (h_running_style_ex(i) == STYLE_EX_OONIGE) return p_pk_speed_up_end_diff_oonige();
    if (pk_pacemaker() == i && h_running_style(i) != STYLE_NIGE) return p_pk_speed_up_end_diff();
    int n = 0;
    for (int k = 0; k < r_style_len(uint(STYLE_NIGE)); k++) {
        if (h_running_style_ex(uint(r_style_horse(uint(STYLE_NIGE) * MAX_HORSES + uint(k)))) == STYLE_EX_NONE) n++;
    }
    if (n == 1 && !r_pre_2026_07()) return p_pk_speed_up_end_diff_one();
    return p_pk_speed_up_end_diff();
}

// Already far enough ahead of the horse behind (false when there is none).
bool pk_check_distance_diff_behind(uint i) {
    int b = horse_at_order(h_cur_order(i) + 1);
    if (b < 0) return false;
    float gap = h_distance(i) - h_distance(uint(b));
    return pk_speed_up_end_distance_diff(i) <= gap;
}

float pk_pacemaker_distance_diff(uint i) {
    uint pm = pk_pacemaker();
    return h_distance(pm) - h_distance(i);
}

bool pk_has_speed_up_skill_value(uint i) {
    return apply_modifier(i, MOD_CURRENT_SPEED, 0.0) > 0.0 || apply_modifier(i, MOD_TARGET_SPEED, 0.0) > 0.0;
}

// Tries to start mode `idx`; all of position keep's RNG happens here.
bool pk_try_start(uint i, int idx, bool with_wiz_roll) {
    float dist = h_distance(i);
    bool oonige = h_running_style_ex(i) == STYLE_EX_OONIGE;
    float end = dist + (idx == PK_NULL ? 0.0 : h_pk_continue_distance(i));
    if (idx == PK_NULL) return false;
    if (idx == PK_SPEED_UP) {
        if (pk_check_distance_diff_behind(i)) return false;
        float v = oonige ? p_pk_speed_up_per_val1_oonige() : p_pk_speed_up_per_val1();
        if (with_wiz_roll && !pk_wiz_roll(i, v)) return false;
    } else if (idx == PK_OVERTAKE) {
        float v = oonige ? p_pk_overtake_per_val1_oonige() : p_pk_overtake_per_val1();
        if (with_wiz_roll && !pk_wiz_roll(i, v)) return false;
    } else if (idx == PK_PACE_UP) {
        if (with_wiz_roll && !pk_wiz_roll(i, p_pk_pace_up_per_val1())) return false;
        vec2 band = pk_offset_distance(r_course_distance(), h_pk_style(i));
        set_h_pk_ps_min(i, 0u, band.x);
        set_h_pk_ps_max(i, 0u, band.y);
        set_h_pk_ps_target(i, 0u, race_random_float_range(0u, band.x, band.y));
    } else if (idx == PK_PACE_DOWN) {
        if (pk_has_speed_up_skill_value(i)) return false;
        bool mid = h_phase(i) == PHASE_MIDDLE;
        vec2 band = pk_offset_distance(r_course_distance(), h_pk_style(i));
        bool post_july = !r_pre_2026_07();
        float mx = band.y;
        if (mid && post_july) mx = float((double(mx) - double(band.x)) * double(mathf_clamp01(p_pk_diff_max_coef())) + double(band.x));
        set_h_pk_ps_min(i, 1u, band.x);
        set_h_pk_ps_max(i, 1u, mx);
        set_h_pk_ps_target(i, 1u, race_random_float_range(0u, band.x, mx));
        set_h_pk_mult(i, 4u, mid && post_july ? p_pk_pace_down_mult_middle() : p_pk_pace_down_mult());
    }
    set_h_pk_end_distance(i, uint(idx), end);
    return true;
}

bool pk_check_end(uint i, int idx) {
    float dist = h_distance(i);
    float end = h_pk_end_distance(i, uint(idx));
    if (idx == PK_NULL) return true;
    if (idx == PK_SPEED_UP) return pk_check_distance_diff_behind(i) || dist >= end;
    if (idx == PK_OVERTAKE) {
        if (end <= dist) return true;
        // Also ends once first overall and the end distance clear of second place.
        if (h_cur_order(i) > 0) return false;
        int second = horse_at_order(1);
        if (second < 0) return true;
        float diff = h_running_style_ex(i) == STYLE_EX_OONIGE ? p_pk_overtake_end_diff_oonige() : p_pk_overtake_end_diff();
        return !(diff > dist - h_distance(uint(second)));
    }
    if (idx == PK_PACE_UP) {
        if (dist >= end) return true;
        return pk_pacemaker_distance_diff(i) <= h_pk_ps_target(i, 0u);
    }
    if (idx == PK_PACE_DOWN) {
        if (!r_pre_2026_07() && r_pm_horse() == int(i)) return true;
        if (pk_has_speed_up_skill_value(i) || end <= dist) return true;
        return pk_pacemaker_distance_diff(i) >= h_pk_ps_target(i, 1u);
    }
    if (end <= dist) return true;
    return !pk_backward_style_ahead(h_cur_order(i), h_running_style(i));
}

// Inside the check window, after the cooldown and check interval: pick a mode and try to start it.
void pk_check_start(uint i, float dt) {
    float d = h_distance(i);
    if (d < h_pk_check_start(i) || h_pk_check_end(i) <= d) return;
    if (h_pk_cool_down(i) > 0.0) set_h_pk_cool_down(i, h_pk_cool_down(i) - dt);
    if (h_pk_cool_down(i) > 0.0) return;
    if (h_pk_check_interval(i) > 0.0) set_h_pk_check_interval(i, h_pk_check_interval(i) - dt);
    if (h_pk_check_interval(i) > 0.0) return;
    set_h_pk_check_interval(i, float(p_pk_check_interval_sec()));
    int mode = pk_calc_keep_mode(i);
    int idx = mode >= 0 && mode < 6 ? mode : PK_NULL;
    bool with_wiz_roll = !h_is_temptation(i);
    if (pk_try_start(i, idx, with_wiz_roll)) {
        set_h_pk_cur(i, idx);
        set_h_position_keep_count(i, h_position_keep_count(i) + 1);
    }
}

void position_keep_update(uint i, float dt) {
    pk_check_temptation(i);
    if (pk_mode(i) == PK_NULL) {
        pk_check_start(i, dt);
        return;
    }
    if (pk_check_end(i, h_pk_cur(i))) pk_clear(i);
}

#endif
