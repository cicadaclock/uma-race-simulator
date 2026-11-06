// ai/overtake.rs: the overtake targets, the overtake cooldown and the overtake-target timers. Needs exact.glsl,
// race_state.glsl, modifiers.glsl and queries.glsl.

#ifndef UMA_OVERTAKE_GLSL
#define UMA_OVERTAKE_GLSL

// speed::target_speed_with_skill: the AI target speed with the CurrentSpeed modifier.
float target_speed_with_skill(uint h) { return apply_modifier(h, MOD_CURRENT_SPEED, h_target_speed(h)); }

bool ot_contains(uint h, int j) {
    for (int k = 0; k < h_ot_count(h); k++) {
        if (h_ot_targets(h, uint(k)) == j) return true;
    }
    return false;
}

void ot_push(uint h, int j) {
    int n = h_ot_count(h);
    if (n >= int(MAX_HORSES)) {
        gpu_fault(FAULT_UPDATE_HORSES);
        return;
    }
    set_h_ot_targets(h, uint(n), j);
    set_h_ot_count(h, n + 1);
}

// Whether horse j is worth overtaking. Gaps come from the start-of-frame scan; speeds are live.
bool ot_check_target(uint i, uint j) {
    float distance_gap = h_ar_distance_gap(i, j);
    if (distance_gap < -1.0) return false;
    float other_speed = h_last_speed(j);
    if (h_block_horse(i) == int(j)) return other_speed < target_speed_with_skill(i);
    float speed_gap = h_ar_speed_gap(i, j);
    if (distance_gap < 1.0 || distance_gap > 20.0 || speed_gap < 0.0) return false;
    if (div_rn32(distance_gap, speed_gap) > p_overtake_dist_per_speed()) return false;
    if (h_block_horse(j) < 0) return target_speed_with_skill(j) < target_speed_with_skill(i);
    return other_speed < target_speed_with_skill(i);
}

// Adds new visible targets, drops the ones that no longer qualify, and always keeps the front blocker.
void ot_collect(uint i) {
    for (int k = 0; k < h_visible_horse_count(i); k++) {
        int v = h_visible_horses(i, uint(k));
        if (ot_check_target(i, uint(v)) && !ot_contains(i, v)) ot_push(i, v);
    }
    int n = h_ot_count(i), w = 0;
    for (int k = 0; k < n; k++) {
        int t = h_ot_targets(i, uint(k));
        if (ot_check_target(i, uint(t))) {
            set_h_ot_targets(i, uint(w), t);
            w++;
        }
    }
    for (int k = w; k < n; k++) set_h_ot_targets(i, uint(k), 0);
    set_h_ot_count(i, w);
    int b = h_block_horse(i);
    if (b >= 0 && !ot_contains(i, b)) ot_push(i, b);
}

// While there are targets the cooldown is held full; once they are gone it runs down, and on expiry forces one
// target-lane re-check. Runs after the goal too.
void update_overtake(uint i, float dt) {
    ot_collect(i);
    if (h_ot_count(i) == 0) {
        float cd = h_ot_cool_down(i);
        if (cd > 0.0) {
            cd -= dt;
            set_h_ot_cool_down(i, cd);
            if (0.0 >= cd) set_force_check_next_lane_one_time(i, true);
        }
    } else {
        set_h_ot_cool_down(i, p_overtake_cool_down_time());
    }
}

// How long a horse behind in order has had this horse as an overtake target.
void update_overtake_target_time(uint i, float dt) {
    int my_order = h_cur_order(i);
    bool targeted = false;
    for (uint o = 0u; o < uint(r_num_horses()); o++) {
        if (my_order < h_cur_order(o) && ot_contains(o, int(i))) {
            targeted = true;
            break;
        }
    }
    set_h_overtake_target_time(i, targeted ? h_overtake_target_time(i) + dt : 0.0);
}

// How long this horse has had any overtake target.
void update_overtake_target_have_time(uint i, float dt) {
    if (h_ot_count(i) == 0) {
        set_h_no_order_up_time(i, 0.0);
        set_h_no_order_down_time(i, 0.0);
    } else {
        set_h_no_order_up_time(i, h_no_order_up_time(i) + dt);
        set_h_no_order_down_time(i, h_no_order_down_time(i) + dt);
    }
}

#endif
