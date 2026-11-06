// Stage::EndOfFrame (src/sim/field.rs). Not ported yet: the pacemaker update of the pre-July-2026 courses. Needs
// exact.glsl, race_state.glsl, sort.glsl, queries.glsl, skill_layout.glsl and skill_state.glsl (the skill caches).

#ifndef UMA_END_OF_FRAME_GLSL
#define UMA_END_OF_FRAME_GLSL

// field::update_last_order_time
void update_last_order_time(float dt) {
    int n = r_num_horses();
    uint idx = uint((n - 1 >= 0 && n - 1 < n) ? r_order_horse(uint(n - 1)) : 0);
    set_h_last_order_time(idx, h_last_order_time(idx) + dt);
}

// field::update_running_style_first_and_last_order_horses: each style's list re-sorted by current order, in place.
void update_running_style_first_and_last_order_horses() {
    sort_mode = SORT_ORDER;
    for (uint s = 1u; s < 5u; s++) {
        int len = r_style_len(s);
        for (int k = 0; k < len; k++) sort_h[k] = r_style_horse(s * MAX_HORSES + uint(k));
        if (!intro_sort(len)) set_r_gpu_fault(r_gpu_fault() | 4u);
        for (int k = 0; k < len; k++) set_r_style_horse(s * MAX_HORSES + uint(k), sort_h[k]);
        set_r_style_top_horse(s, len > 0 ? sort_h[0] : -1);
        set_r_style_last_horse(s, len > 0 ? sort_h[len - 1] : -1);
    }
}

// field::cache_rushed_horses: per style and overall, style by style, each in current order. Entries past the new
// lengths are zeroed (the CPU's lists shrink).
void cache_rushed_horses() {
    int all = 0;
    for (uint s = 1u; s < 5u; s++) {
        int len = 0;
        for (int k = 0; k < r_style_len(s); k++) {
            int h = r_style_horse(s * MAX_HORSES + uint(k));
            if (h_is_temptation(uint(h))) {
                set_r_style_rushed_horse(s * MAX_HORSES + uint(len), h);
                len++;
                set_r_rushed_horse(uint(all), h);
                all++;
            }
        }
        for (int k = len; k < int(MAX_HORSES); k++) set_r_style_rushed_horse(s * MAX_HORSES + uint(k), 0);
        set_r_style_rushed_len(s, len);
    }
    for (int k = all; k < int(MAX_HORSES); k++) set_r_rushed_horse(uint(k), 0);
    set_r_rushed_len(all);
}

// field::update_lane_distance_rate: each running horse's lane as a percentage of the field's lane spread.
void update_lane_distance_rate() {
    int n = r_num_horses();
    float mn = uintBitsToFloat(0x7f7fffffu), mx = uintBitsToFloat(0xff7fffffu);  // f32::MAX, -f32::MAX
    for (int i = 0; i < n; i++) {
        if (h_is_finished(uint(i))) continue;
        float l = h_lane_distance(uint(i));
        if (mn > l) mn = l;
        if (l > mx) mx = l;
    }
    float range = mx - mn;
    if (mathf_approximately(0.0, range)) range = 1.0;
    for (int i = 0; i < n; i++) {
        if (h_is_finished(uint(i))) continue;
        double v = div_rn64(double(h_lane_distance(uint(i))) - double(mn), double(range)) * 100.0lf;
        set_h_lane_distance_per_in_all_horses(uint(i), float(v));
    }
}

void end_of_frame(float dt) {
    skills_begin();
    update_last_order_time(dt);
    update_running_style_first_and_last_order_horses();
    cache_rushed_horses();
    cache_skill_lists();
    update_lane_distance_rate();
}

#endif
