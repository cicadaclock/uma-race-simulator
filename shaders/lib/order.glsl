// src/sim/ai/order.rs: the field's order and the order-change counters (Stage::Order before the last frame).
// Needs exact.glsl, race_state.glsl and sort.glsl.

#ifndef UMA_ORDER_GLSL
#define UMA_ORDER_GLSL

#include "queries.glsl"

void update_horse_order() {
    int n = r_num_horses();
    for (int k = 0; k < n; k++) sort_h[k] = r_order_horse(uint(k));
    sort_mode = SORT_CUR_ORDER;
    if (!intro_sort(n)) set_r_gpu_fault(r_gpu_fault() | 1u);
    for (int o = 0; o < n; o++) {
        set_r_order_horse(uint(o), sort_h[o]);
        set_r_order_distance(uint(o), h_distance(uint(sort_h[o])));
        uint h = uint(sort_h[o]);
        set_h_prev_order(h, h_cur_order(h));
        set_h_cur_order(h, o);
    }
}

// Accessors over the up / down counter fields.
float occ_cooldown(uint h, bool up, uint k) { return up ? h_ocu_cooldown(h, k) : h_ocd_cooldown(h, k); }
void set_occ_cooldown(uint h, bool up, uint k, float v) { if (up) set_h_ocu_cooldown(h, k, v); else set_h_ocd_cooldown(h, k, v); }
void occ_add(uint h, bool up, int which) {
    // which: 0 phase_start, 1 phase_middle, 2 phase_end_after, 3 corner, 4 corner_phase_end_after, 5 last_spurt,
    // 6 distance1, 7 final_corner_after
    if (up) {
        if (which == 0) set_h_ocu_phase_start(h, h_ocu_phase_start(h) + 1);
        else if (which == 1) set_h_ocu_phase_middle(h, h_ocu_phase_middle(h) + 1);
        else if (which == 2) set_h_ocu_phase_end_after(h, h_ocu_phase_end_after(h) + 1);
        else if (which == 3) set_h_ocu_corner(h, h_ocu_corner(h) + 1);
        else if (which == 4) set_h_ocu_corner_phase_end_after(h, h_ocu_corner_phase_end_after(h) + 1);
        else if (which == 5) set_h_ocu_last_spurt(h, h_ocu_last_spurt(h) + 1);
        else if (which == 6) set_h_ocu_distance1(h, h_ocu_distance1(h) + 1);
        else set_h_ocu_final_corner_after(h, h_ocu_final_corner_after(h) + 1);
    } else {
        if (which == 0) set_h_ocd_phase_start(h, h_ocd_phase_start(h) + 1);
        else if (which == 1) set_h_ocd_phase_middle(h, h_ocd_phase_middle(h) + 1);
        else if (which == 2) set_h_ocd_phase_end_after(h, h_ocd_phase_end_after(h) + 1);
        else if (which == 3) set_h_ocd_corner(h, h_ocd_corner(h) + 1);
        else if (which == 4) set_h_ocd_corner_phase_end_after(h, h_ocd_corner_phase_end_after(h) + 1);
        else if (which == 5) set_h_ocd_last_spurt(h, h_ocd_last_spurt(h) + 1);
        else if (which == 6) set_h_ocd_distance1(h, h_ocd_distance1(h) + 1);
        else set_h_ocd_final_corner_after(h, h_ocd_final_corner_after(h) + 1);
    }
}

void order_change_counter_update(uint i, bool up, float dt) {
    if (h_is_finished(i)) return;
    bool last_spurt = is_last_spurt(i);
    int cur = h_cur_order(i), prev = h_prev_order(i);
    int phase = h_phase(i), cur_corner = h_cur_corner(i);
    float distance = h_distance(i);
    bool is_final_corner = h_is_final_corner(i);
    bool has_target = up ? prev > cur : cur > prev;
    int target = has_target ? horse_index_at_order(up ? cur + 1 : cur - 1) : 0;
    for (uint k = 0u; k < MAX_HORSES; k++) {
        float cd = occ_cooldown(i, up, k) - dt;
        if (0.0 > cd) cd = 0.0;
        set_occ_cooldown(i, up, k, cd);
    }
    if (up) set_h_ocu_changed(i, false); else set_h_ocd_changed(i, false);
    if (!has_target) return;
    if (!(0.0 >= occ_cooldown(i, up, uint(target)))) return;
    set_occ_cooldown(i, up, uint(target), up ? h_ocu_cool_down_time(i) : h_ocd_cool_down_time(i));
    if (up) set_h_ocu_changed(i, true); else set_h_ocd_changed(i, true);
    if (phase == 0) occ_add(i, up, 0);
    else if (phase == 1) occ_add(i, up, 1);
    else if (phase == 2 || phase == 3) occ_add(i, up, 2);
    if (cur_corner != 0) {
        occ_add(i, up, 3);
        if (phase == 2 || phase == 3) occ_add(i, up, 4);
    }
    if (last_spurt) occ_add(i, up, 5);
    if (!(200.0 > distance)) occ_add(i, up, 6);
    if (is_final_corner) occ_add(i, up, 7);
    if (up) set_h_no_order_up_time(i, 0.0);
    else set_h_no_order_down_time(i, 0.0);
}

// order::calc_horse_order
void calc_horse_order(float dt) {
    update_horse_order();
    for (int i = 0; i < r_num_horses(); i++) {
        order_change_counter_update(uint(i), true, dt);
        order_change_counter_update(uint(i), false, dt);
    }
}

#endif
