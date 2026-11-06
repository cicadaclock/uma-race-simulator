// physics/slope.rs and physics/conserve_power.rs. Entering a slope is a course event (end of step, not ported
// yet); the conserve-power release event is output only. Needs exact.glsl, race_state.glsl, net_random.glsl,
// modifiers.glsl, queries.glsl and stats.glsl.

#ifndef UMA_CONSERVE_POWER_GLSL
#define UMA_CONSERVE_POWER_GLSL

// Leaves the slope once past its end (right after the distance step).
void update_slope(uint i) {
    if (h_slope_type(i) != SLOPE_NULL && h_slope_end_distance(i) < h_distance(i)) {
        set_h_slope_type(i, SLOPE_NULL);
        set_h_slope_per(i, 0.0);
    }
}

// Enters / leaves downhill accel mode, checked once per second while on a downhill (also when finished). One
// GetRandom(0f, 100f) per check.
void update_down_slope_accel(uint i, float dt) {
    bool on_down = h_slope_type(i) == SLOPE_DOWN;
    if (!h_down_slope_mode(i)) {
        if (!on_down) return;  // the timer is not reset
        float c = h_down_slope_check_sec(i);
        if (c > 0.0) {
            set_h_down_slope_check_sec(i, c - dt);
            return;
        }
        set_h_down_slope_check_sec(i, 1.0);
        float per = mathf_min(100.0, stat_wiz_for(i, WIZ_DOWNHILL) * p_down_slope_start_wiz_rate());
        float r = race_random_float_range(0u, 0.0, 100.0);
        if (r < per) {
            set_h_down_slope_mode(i, true);
            set_h_down_slope_check_sec(i, 1.0);
        }
    } else {
        if (!on_down) {
            set_h_down_slope_mode(i, false);
            set_h_down_slope_check_sec(i, 1.0);
            return;
        }
        float c = h_down_slope_check_sec(i);
        if (c > 0.0) {
            set_h_down_slope_check_sec(i, c - dt);
            return;
        }
        set_h_down_slope_check_sec(i, 1.0);
        float r = race_random_float_range(0u, 0.0, 100.0);
        if (r < p_down_slope_end_per()) {
            set_h_down_slope_mode(i, false);
            set_h_down_slope_check_sec(i, 1.0);
        }
    }
}

// The lookup tables of conserve_power.rs coef_by, by value (1 when outside the packed range, as for a missing row).
float cp_release_time_distance_coef(int dtype) { return dtype >= 0 && dtype < 5 ? p_cp_release_time_distance_coef(uint(dtype)) : 1.0; }
float cp_style_coef(int style) { return style >= 0 && style < 5 ? p_cp_style_coef(uint(style)) : 1.0; }
float cp_release_accel_coef_by(int style, int dtype) {
    return style >= 0 && style < 5 && dtype >= 0 && dtype < 5 ? p_cp_release_accel_coef_by(uint(style * 5 + dtype)) : 1.0;
}

// Conserve power (not when finished): accrue in whole seconds while running relaxed before the position-keep end,
// release on the first last-spurt frame.
void conserve_power_update(uint i, float dt) {
    if (!h_cp_active(i)) return;
    if (!h_cp_released(i)) {
        if (is_last_spurt(i)) {
            set_h_cp_released(i, true);
            int dtype = r_course_distance_type();
            float dcoef = cp_release_time_distance_coef(dtype);
            float dist = float(r_course_distance());
            float duration = div_rn32(h_conserved_power(i) * sqrt_rn32(dist) * dcoef, p_cp_release_time_coef());
            if (duration >= p_cp_releasable_time_threshold()) {
                float scoef = cp_release_accel_coef_by(h_running_style(i), dtype);
                set_h_cp_accel(i, sqrt_rn32((h_cp_power_stat(i) - 1200.0) * p_cp_release_decl_coef()) * p_cp_release_accel_coef() * scoef * h_conserve_activity_coef(i));
                set_h_cp_remain_time(i, duration + dt);
            }
        } else {
            int mode = pk_mode(i);
            bool tempt = h_is_temptation(i);
            // Entering rushed (decrease mode 4; the spot struggle's mode 3 is in Field).
            if (tempt && !h_cp_temptation_prev(i)) {
                set_h_conserved_power(i, h_conserved_power(i) * p_cp_decrease_coef_temptation());
                set_h_conserve_activity_coef(i, h_conserve_activity_coef(i) * p_cp_decrease_accel_coef_temptation());
            }
            set_h_cp_temptation_prev(i, tempt);
            bool blocked = tempt || h_is_compete_top(i) || !(mode == PK_NULL || mode == PK_PACE_DOWN);
            if (blocked) {
                set_h_cp_cool_down(i, p_cp_default_cool_down());
                set_h_cp_bank_timer(i, 0.0);
            } else {
                float cd = h_cp_cool_down(i);
                if (cd > 0.0) {
                    cd -= dt;
                    set_h_cp_cool_down(i, cd);
                }
                if (cd > 0.0 || h_distance(i) >= h_pk_check_end(i)) return;
                float bank = h_cp_bank_timer(i) + dt;
                set_h_cp_bank_timer(i, bank);
                if (bank >= 1.0) {
                    set_h_cp_bank_timer(i, bank - 1.0);
                    // Increase mode: 1 = pace down, 2 = no position keep.
                    float k = mode == PK_PACE_DOWN ? p_cp_increase_coef_pace_down() : p_cp_increase_coef_normal();
                    float s = cp_style_coef(h_running_style(i));
                    set_h_conserved_power(i, h_conserved_power(i) + k * s);
                }
            }
        }
    }
    float rt = h_cp_remain_time(i);
    if (rt > 0.0) {
        rt -= dt;
        set_h_cp_remain_time(i, rt);
        if (rt <= 0.0) set_h_cp_accel(i, 0.0);
    }
}

#endif
