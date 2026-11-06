// Stage::UpdateHorses (tick.rs update_horse, advance, run_ai, ai_decisions): decisions and movement per horse, in
// horse order (later horses see earlier horses' new state). Not here: the conserve-power release event (output
// only). Needs every file of stage_update_horses.comp.

#ifndef UMA_UPDATE_HORSES_GLSL
#define UMA_UPDATE_HORSES_GLSL

// Called at the end of the AI step, before the move is applied.
void update_phase(uint i) {
    float d = h_distance(i);
    int ph;
    if (d < r_phase_middle_start()) ph = PHASE_START;
    else if (d < r_phase_end_start()) ph = PHASE_MIDDLE;
    else if (d < r_phase_last_start()) ph = PHASE_END;
    else if (d >= r_phase_course_distance()) ph = PHASE_FINISHED;
    else ph = PHASE_LAST;
    set_h_phase(i, ph);
}

void ai_decisions(uint i, float dt) {
    if (!h_is_finished(i)) position_keep_update(i, dt);
    update_overtake(i, dt);
    // Rush (a skill may set its end time first).
    if (!h_is_finished(i)) {
        if (h_is_temptation(i)) {
            float t = apply_modifier(i, MOD_TEMPTATION_END_TIME, 0.0);
            if (!mathf_approximately(t, 0.0)) {
                set_h_tm_skill_end_time(i, t);
                set_h_tm_skill_end_enable(i, true);
            }
        }
        temptation_update(i, dt);
    }
    update_move_lane(i);
    update_force_in_move(i);
    if (!h_ls_by_distance_requested(i) && h_phase(i) > PHASE_MIDDLE) {
        set_h_req_calc_ls(i, true);
        set_h_ls_by_distance_requested(i, true);
    }
    update_last_spurt(i);
    update_down_slope_accel(i, dt);
    if (!h_is_finished(i)) conserve_power_update(i, dt);
    update_target_speed(i);
}

// Returns the distance moved.
float run_ai(uint i, float dt) {
    if (mathf_approximately(dt, 0.0)) return 0.0;
    set_h_lane_move_speed(i, 0.0);
    if (h_delay_time(i) > 0.0) {
        float d = h_delay_time(i) - dt;
        set_h_delay_time(i, d);
        if (d > 0.0) return 0.0;
    }
    ai_decisions(i, dt);
    float max_speed = calc_max_speed(i);
    move_lane(i, dt);
    float self_speed = update_self_speed(i, dt, max_speed);
    float current_speed_mod = apply_modifier(i, MOD_CURRENT_SPEED, 0.0);
    float natural_decel_mod = apply_modifier(i, MOD_CURRENT_SPEED_WITH_NATURAL_DECELERATION, 0.0);
    float v = float((double(natural_decel_mod) + double(current_speed_mod)) + double(self_speed));
    float min_speed = h_min_speed(i);
    if (min_speed > v) v = min_speed;
    else if (v > max_speed) v = max_speed;
    float moved = v * dt;
    set_h_last_speed(i, v > min_speed ? v : min_speed);
    if (0.0 > h_delay_time(i)) {
        // The gate opened mid-frame: move only for the part after it.
        moved = float(double(moved) * div_rn64(double(abs(h_delay_time(i))), double(dt)));
        set_h_delay_time(i, 0.0);
    }
    if (!h_is_finished(i)) {
        consume_hp(i, dt, self_speed);
        if (h_hp(i) <= 0.0) set_h_is_hp_empty_on_race(i, true);
    }
    update_phase(i);
    set_h_prev_elapsed_time(i, dt);
    return moved;
}

// `distance` advances by at most the world-space displacement of the move.
void advance(uint i, float dt) {
    set_h_prev_distance(i, h_distance(i));
    for (uint k = 0u; k < 3u; k++) set_h_prev_position(i, k, h_position(i, k));
    float moved = run_ai(i, dt);
    vec3 w = world_position(h_distance(i), h_lane_distance(i));
    set_h_position(i, 0u, w.x);
    set_h_position(i, 1u, w.y);
    set_h_position(i, 2u, w.z);
    float dx = h_prev_position(i, 0u) - w.x;
    float dz = h_prev_position(i, 2u) - w.z;
    float world_disp = float(sqrt_rn64((double(dx) * double(dx) + 0.0lf) + double(dz) * double(dz)));
    float lim = moved * r_world_per_metre();
    if (lim < world_disp) moved = float(double(moved) * div_rn64(double(lim), double(world_disp)));
    set_h_distance(i, h_distance(i) + moved);
    update_slope(i);
    // The start dash ends once the self speed reaches the default minimum speed.
    if (h_is_start_dash(i) && h_race_default_min_speed(i) <= h_last_self_speed(i)) set_h_is_start_dash(i, false);
}

// Front / side block timers, from the start-of-frame scan.
void update_block_time(uint i, float dt) {
    int front = h_block_horse(i);
    bool front_hp_empty = front >= 0 && h_hp(uint(front)) <= 0.0;
    bool side = h_block_horse_in(i) >= 0 || h_block_horse_out(i) >= 0;
    if (front >= 0) set_h_block_front_accumulate_time(i, h_block_front_accumulate_time(i) + dt);
    set_h_block_front_continue_time(i, front >= 0 ? h_block_front_continue_time(i) + dt : 0.0);
    set_h_block_front_hp_empty_time(i, front_hp_empty ? h_block_front_hp_empty_time(i) + dt : 0.0);
    if (side) {
        float t = h_block_side_continue_time(i) + dt;
        set_h_block_side_continue_time(i, t);
        // Per-phase maximum; Finished has no entry.
        int ph = h_phase(i);
        if (ph >= 0 && ph < 4 && h_block_side_max_by_phase(i, uint(ph)) < t) set_h_block_side_max_by_phase(i, uint(ph), t);
    } else {
        set_h_block_side_continue_time(i, 0.0);
    }
}

// trigger::order_in_out: the (10 * (k + 1))% threshold of n horses, as rescale_rate.
int order_in_out_th(int k, int n) { return f32_to_i32(div_rn32(float(10 * (k + 1)), 100.0) * float(n) + 0.5); }

// Clears each order-in (order-out) flag once the horse falls behind (gets ahead of) that threshold, from 5 s in.
void update_order_in_out_continue(uint i) {
    if (r_accumulate_time() < 5.0) return;
    int n = r_num_race_horses();
    int place = h_cur_order(i) + 1;
    for (int k = 0; k < int(ORDER_IN_OUT_TYPES); k++) {
        if (h_order_in_continue(i, uint(k)) && order_in_out_th(k, n) < place) set_h_order_in_continue(i, uint(k), false);
    }
    for (int k = 0; k < int(ORDER_IN_OUT_TYPES); k++) {
        if (h_order_out_continue(i, uint(k)) && place < order_in_out_th(k, n)) set_h_order_out_continue(i, uint(k), false);
    }
}

// Finish time, interpolated back to the moment the horse crossed the line.
void calc_finish_time(uint i) {
    if (h_finish_time_scaled(i) >= 0.0) return;
    float cd = float(r_course_distance());
    float t = mathf_clamp(div_rn32(h_distance(i) - cd, h_distance(i) - h_prev_distance(i)), 0.0, 1.0);
    float raw = float(double(r_accumulate_time()) - double(t) * double(h_prev_elapsed_time(i)));
    set_h_finish_time_raw(i, raw);
    double scaled = double(raw) * double(p_finish_time_coef());
    set_h_finish_time_for_order(i, scaled);
    set_h_finish_time_scaled(i, float(scaled));
}

void check_finished(uint i) {
    if (!h_is_finished(i) && float(r_course_distance()) < h_distance(i)) {
        set_h_is_finished(i, true);
        if (h_finish_time_scaled(i) < 0.0) calc_finish_time(i);
        clear_skills(i);
    }
}

bool update_horse(uint i, float dt) {
    update_min_speed(i);
    update_hp_by_skill(i);
    advance(i, dt);
    // The near-around timer runs before the overtake timers.
    if (h_near_horse_count(i) >= 4) set_h_near_around_time(i, h_near_around_time(i) + dt);
    update_overtake_target_time(i, dt);
    update_overtake_target_have_time(i, dt);
    update_block_time(i, dt);
    bool ls = is_last_spurt(i);
    if (!h_is_last_spurt_prev(i) && ls) set_h_last_spurt_order(i, h_cur_order(i));
    set_h_is_last_spurt_prev(i, ls);
    if (h_final_corner_end_order(i) == -1 && h_is_final_corner(i) && h_cur_corner(i) == 0) set_h_final_corner_end_order(i, h_cur_order(i));
    update_order_in_out_continue(i);
    check_finished(i);
    return h_is_finished(i);
}

// Every horse is updated (no short-circuit); sets frame_all_finished.
void update_horses_stage(float dt) {
    skills_begin();
    bool all_finished = true;
    for (uint i = 0u; i < uint(r_num_horses()); i++) {
        bool f = update_horse(i, dt);
        all_finished = all_finished && f;
    }
    set_r_frame_all_finished(all_finished);
}

#endif
