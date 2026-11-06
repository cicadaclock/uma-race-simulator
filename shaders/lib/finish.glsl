// The last frame of Stage::Order: field::calc_finish_order, field::clamp_finish_time and horse::create_defeat.
// Needs exact.glsl, race_state.glsl and sort.glsl.

#ifndef UMA_FINISH_GLSL
#define UMA_FINISH_GLSL

void calc_finish_order() {
    if (r_is_finish_order_calculated()) return;
    set_r_is_finish_order_calculated(true);
    int n = r_num_horses();
    for (int k = 0; k < n; k++) sort_h[k] = k;
    sort_mode = SORT_FINISH_TIME;
    if (!intro_sort(n)) set_r_gpu_fault(r_gpu_fault() | 2u);
    for (int order = 0; order < n; order++) {
        uint i = uint(sort_h[order]);
        set_r_finish_order_horse(uint(order), int(i));
        set_h_finish_order(i, order);
        if (order > 0) {
            float prev = h_finish_time_scaled(uint(sort_h[order - 1]));
            set_h_finish_time_diff_from_prev(i, h_finish_time_scaled(i) - prev);
        }
    }
}

// The winner's time into the course set's range; every horse shifts by the same amount. The clamped values (a
// fresh RaceRandom(seed)'s first draw) come precomputed. NaN clamps (to the high side).
void clamp_finish_time() {
    uint first = uint(r_finish_order_horse(0u));
    float t0 = h_finish_time_scaled(first);
    float mn = r_finish_time_min(), mx = r_finish_time_max();
    if (t0 >= mn && mx >= t0) return;
    float t = mn > t0 ? r_finish_time_clamp_low() : r_finish_time_clamp_high();
    float diff = t - h_finish_time_scaled(first);
    set_h_finish_time_scaled(first, t);
    for (int k = 1; k < r_num_horses(); k++) {
        uint i = uint(r_finish_order_horse(uint(k)));
        set_h_finish_time_scaled(i, h_finish_time_scaled(i) + diff);
    }
}

// Rounded average skill count of the top three finishers other than `except` (0 when none).
int calc_top_passive_skill_num_average(uint except) {
    int sum = 0, count = 0;
    for (int k = 0; k < 3 && k < r_num_horses(); k++) {
        uint h = uint(r_finish_order_horse(uint(k)));
        if (h != except) {
            sum += h_skill_count(h);
            count++;
        }
    }
    if (count == 0) return 0;
    return f32_to_i32(div_rn32(float(sum), float(count)) + 0.5);
}

// horse::calc_defeat: the first rule that applies (horse::Defeat discriminants).
int calc_defeat(uint i) {
    float n = float(r_num_race_horses());
    int fo = h_finish_order(i);
    if (fo < 1) return 1;
    if (fo <= p_defeat_lose_finish_order() - 1) {
        float x = h_finish_time_diff_from_prev(i) * 15.0;
        x = 0.0 > x ? 0.0 : div_rn32(x, 2.5);
        if (floor_to_int(x) <= p_defeat_lose_bashin_diff()) return 2;
    }
    if (h_active_proper_distance(i) < p_defeat_proper_distance()) return 12;
    if (h_active_proper_ground_type(i) < p_defeat_proper_ground()) return 13;
    int style_count = r_style_count(uint(h_running_style(i)));
    if (style_count > f32_to_i32(div_rn32(float(p_defeat_same_running_style_per()), 100.0) * n + 0.5)) return 3;
    if (h_is_hp_empty_on_race(i)) {
        if (h_temptation_count(i) >= p_defeat_temptation_count()) return 4;
        int threshold = f32_to_i32(div_rn32(float(p_defeat_guts_order_per()), 100.0) * n + 0.5);
        return h_guts_order(i) + 1 < threshold ? 6 : 5;
    }
    int lsr = h_ls_calc_result(i);
    if ((lsr & 12) != 0) return 7;  // FALSE_FLAGS
    if (lsr == 2) return 8;  // TRUE
    if (h_motivation(i) <= p_defeat_motivation()) return 14;
    if (h_skill_count(i) < calc_top_passive_skill_num_average(i)) return 9;
    float b = h_block_front_accumulate_time(i);
    return (b < p_defeat_front_block_accumulate_time() || is_nan32(b)) ? 11 : 10;
}

// tick.rs order_stage, all_finished branch (team scoring is not simulated).
void finish_frame() {
    calc_finish_order();
    clamp_finish_time();
    if (!r_is_defeat_checked()) {
        for (int i = 0; i < r_num_horses(); i++) set_h_defeat(uint(i), calc_defeat(uint(i)));
    }
}

#endif
