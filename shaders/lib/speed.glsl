// physics/speed.rs and physics/hp.rs: min / max speed, target speed, acceleration, the speed update, HP, and the
// spot-struggle / dueling adds (compete_top::add_target_speed, compete_fight::add_target_speed / add_accel). Needs
// exact.glsl, race_state.glsl, modifiers.glsl, queries.glsl, stats.glsl, transcend.glsl and position_keep.glsl.

#ifndef UMA_SPEED_GLSL
#define UMA_SPEED_GLSL

float ct_add_target_speed(uint i) {
    if (!h_is_compete_top(i)) return 0.0;
    return mathf_pow(p_ct_add_param1_coef1() * stat_guts(i), p_ct_add_param1_coef2()) * p_ct_add_param1_coef3();
}
float cf_add_target_speed(uint i) {
    if (!h_is_compete_fight(i)) return 0.0;
    return mathf_pow(p_cf_add_param1_coef1() * stat_guts(i), p_cf_add_param1_coef2()) * p_cf_add_param1_coef3();
}
float cf_add_accel(uint i) {
    if (!h_is_compete_fight(i)) return 0.0;
    return mathf_pow(p_cf_add_param2_coef1() * stat_guts(i), p_cf_add_param2_coef2()) * p_cf_add_param2_coef3();
}

// Never negative.
float calc_accel_per_sec(uint i) {
    if (h_is_finished(i)) return 1.0;
    float pow_coef = h_slope_type(i) == SLOPE_UP ? p_accel_pow_coef_up_slope() : p_accel_pow_coef();
    int ph = h_phase(i);
    float phase_coef = ph >= PHASE_START && ph <= PHASE_LAST ? h_phase_accel_coef(i, uint(ph)) : 1.0;
    float a = sqrt_rn32(stat_pow(i) * p_accel_pow_coef_sqrt());
    a = float((double(a) * double(pow_coef)) * double(h_proper_ground_type_coef(i)) * double(h_proper_distance_coef_pow(i)) * double(phase_coef));
    a = apply_modifier(i, MOD_ACCEL, a);
    a = a + cf_add_accel(i);
    a = a + h_cp_accel(i);
    if (h_is_start_dash(i)) a = a + p_start_accel_add();
    return mathf_max(0.0, a);
}

// Base target speed (with its random part) of the section at the horse's distance.
float base_target_speed(uint i) {
    if (!h_bts_initialized(i)) return r_base_speed();
    // The quotient is truncated from f64.
    int k = f64_to_i32(div_rn64(double(h_distance(i)), double(r_course_section_distance())));
    k = k < 0 ? 0 : min(k, int(BTS_SECTIONS) - 1);
    return h_bts_random(i, uint(k)) + h_bts_speed(i, uint(k));
}

// Base target speed of the first section in `phase`, without the random part.
float base_target_speed_without_random(uint i, int phase) {
    if (!h_bts_initialized(i)) return r_base_speed();
    for (uint k = 0u; k < BTS_SECTIONS; k++) {
        if (h_bts_phase(i, k) == phase) return h_bts_speed(i, k);
    }
    return r_base_speed();
}

// This frame's target speed (the order of the terms is exact).
void calc_target_speed(uint i) {
    float t = is_last_spurt(i) ? h_ls_target_speed(i) : base_target_speed(i);
    if (is_position_keep(i)) t = pk_base_target_speed_multiply(i) * t;
    if (h_force_in_move(i)) t = t + h_force_in_move_add(i);
    t = apply_modifier(i, MOD_TARGET_SPEED, t);
    if (!mathf_approximately(h_lane_distance(i), h_prev_lane_distance(i)) && has_modifier(i, MOD_TARGET_SPEED_ON_MOVE_LANE)) {
        t = t + mathf_pow(stat_pow(i) * p_lane_move_add_param1(), p_lane_move_add_param2());
    }
    float slope = 0.0;
    if (h_slope_type(i) == SLOPE_UP) {
        float pw = stat_pow(i);
        float sp = h_slope_per(i);
        if (sp > 0.0) slope = mathf_min(0.0, float(-(double(abs(sp)) * div_rn64(double(p_up_slope_add_speed_val1()), double(pw)))));
    } else if (h_slope_type(i) == SLOPE_DOWN && h_down_slope_mode(i)) {
        float sp = h_slope_per(i);
        if (sp < 0.0) slope = mathf_max(0.0, float(div_rn64(double(abs(sp)), double(p_down_slope_add_speed_val2())) + double(p_down_slope_add_speed_val1())));
    }
    t = t + slope;
    t = t + ct_add_target_speed(i);
    t = t + cf_add_target_speed(i);
    if (h_hp(i) <= 0.0) t = h_min_speed(i);
    set_h_calc_target_speed(i, mathf_clamp(t, p_target_speed_min(), 30.0));
}

// The last step of the AI update; after the goal the target speed is frozen.
void update_target_speed(uint i) {
    if (!h_is_finished(i)) {
        calc_target_speed(i);
        set_h_target_speed(i, h_calc_target_speed(i));
    }
}

// Min speed from Guts, capped by a blocking horse in front (first step of the horse update).
void update_min_speed(uint i) {
    if (h_is_finished(i)) {
        set_h_min_speed(i, 5.0);
    } else if (h_is_start_dash(i)) {
        set_h_min_speed(i, p_start_speed());
    } else {
        float g = stat_guts(i);
        set_h_min_speed(i, float(double(sqrt_rn32(g * p_min_speed_guts_coef_sqrt())) * double(p_min_speed_guts_coef()) + double(h_race_default_min_speed(i))));
        int fb = h_block_horse(i);
        if (fb >= 0) set_h_min_speed(i, mathf_min(h_min_speed(i), h_min_speed(uint(fb))));
    }
}

// Speed cap while blocked: a fraction of the blocking horse's speed, rising with the gap (30 when not blocked).
float calc_max_speed(uint i) {
    int b = h_block_horse(i);
    if (b < 0) return 30.0;
    float x = div_rn32(h_ar_distance_gap(i, uint(b)), p_front_block_distance_gap());
    x = mathf_clamp01(x);
    x = mathf_clamp01(x);
    float lo = p_front_block_max_speed_rate_min(), hi = p_front_block_max_speed_rate_max();
    float cap = h_last_speed(uint(b)) * float((double(hi) - double(lo)) * double(x) + double(lo));
    return mathf_max(h_min_speed(i), cap);
}

// Moves the speed toward the target (accelerate or decelerate), clamped to [min speed, max_speed].
float update_self_speed(uint i, float dt, float max_speed) {
    float t = h_target_speed(i);
    float v = h_last_self_speed(i);
    v = v + apply_one_time_skill_modifier(i, 0.0);
    if (t > v) {
        float accel = calc_accel_per_sec(i);
        v = mathf_min(t, accel * dt + v);
    } else if (v > t) {
        bool pace_down = is_position_keep(i) && pk_mode(i) == PK_PACE_DOWN;
        float dec;
        if (h_is_finished(i)) {
            dec = 4.0;
        } else {
            int ph = h_phase(i);
            float d = ph == PHASE_START ? p_decl_rate_start() : (ph == PHASE_MIDDLE ? p_decl_rate_middle() : p_decl_rate_end());
            // The PaceDown decel arrived with the July 2026 update.
            if (pace_down && !r_pre_2026_07()) d = p_decl_rate_pace_down();
            if (h_hp(i) <= 0.0 && p_decl_rate_hp_zero() > d) d = p_decl_rate_hp_zero();
            dec = d * p_decl_base();
        }
        v = v - dec * dt;
        if (t > v) v = t;
    }
    if (h_is_start_dash(i)) v = mathf_min(h_race_default_min_speed(i), v);
    v = mathf_clamp(v, h_min_speed(i), max_speed);
    set_h_last_self_speed(i, v);
    return v;
}

// ---- hp.rs

// This frame's skill heal / drain; a heal from the End phase on requests a last-spurt recompute.
void update_hp_by_skill(uint i) {
    float v = apply_modifier(i, MOD_HP, 0.0);
    if (!mathf_approximately(v, 0.0)) {
        set_h_hp(i, mathf_clamp(v + h_hp(i), 0.0, h_max_hp(i)));
        if (v > 0.0 && h_phase(i) >= PHASE_END) set_h_req_calc_ls(i, true);
    }
}

// HP/s at speed v (hp::hp_per_sec). Pow(x, 2) is x * x: the f64 square of an f32 is exact.
float hp_per_sec(float rbs, float rate, float multi_hp_sub, int phase, float guts, float v) {
    float gap = div_rn32(mathf_pow2((v - rbs) + p_hp_speed_gap_param1()), p_hp_speed_gap_param1_pow());
    float x = float(double(gap) * double(p_hp_dec_base()) * double(rate) * double(multi_hp_sub));
    if (phase >= PHASE_END) x = x * float(div_rn64(double(p_hp_guts_coef()), double(sqrt_rn32(guts * p_hp_guts_coef_sqrt()))) + 1.0lf);
    return x;
}

float calc_hp_dec_rate(uint i) {
    bool oonige = h_running_style_ex(i) == STYLE_EX_OONIGE;
    bool tempt = h_is_temptation(i);
    float rate = p_hp_rate_normal();
    float s = apply_modifier(i, MOD_HP_DEC_RATE, 0.0);
    if (s > 0.0) rate = s;
    else if (tempt && h_is_compete_top(i)) rate = oonige ? p_hp_rate_tempt_ct_oonige() : p_hp_rate_tempt_ct_nige();
    else if (h_is_compete_top(i)) rate = oonige ? p_hp_rate_ct_oonige() : p_hp_rate_ct_nige();
    else if (tempt) rate = p_hp_rate_temptation();
    else if (is_position_keep(i) && pk_mode(i) == PK_PACE_DOWN) rate = p_hp_rate_pace_down();
    if (h_down_slope_mode(i)) rate = rate * p_hp_rate_down_slope_multiply();
    return rate;
}

// HP consumption of one frame at self_speed (the caller skips finished horses).
void consume_hp(uint i, float dt, float self_speed) {
    if (h_hp(i) > 0.0) {
        float rbs = h_is_start_dash(i) ? self_speed : r_base_speed();
        float dec = hp_per_sec(rbs, calc_hp_dec_rate(i), h_multi_hp_sub(i), h_phase(i), stat_guts(i), self_speed);
        set_h_hp(i, mathf_clamp(-(dec * dt) + h_hp(i), 0.0, h_max_hp(i)));
    }
}

// HP needed to run `distance` at `target_speed` at the normal rate.
float calc_need_hp(float rbs, float multi_hp_sub, int phase, float guts, float target_speed, float distance) {
    return div_rn32(distance, target_speed) * hp_per_sec(rbs, p_hp_rate_normal(), multi_hp_sub, phase, guts, target_speed);
}

#endif
