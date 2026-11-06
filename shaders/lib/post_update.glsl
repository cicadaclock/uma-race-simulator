// Stage::PostUpdate: the near-horse timers (src/sim/ai/around.rs). debug::step only prints. Needs exact.glsl and
// race_state.glsl.

#ifndef UMA_POST_UPDATE_GLSL
#define UMA_POST_UPDATE_GLSL

uint horse_by_order(int order) { return uint(r_order_horse(uint(order))); }

void update_in_front_horse_near_time(uint i, float dt) {
    int cur = h_cur_order(i), prev = h_prev_order(i);
    if (cur == 0 || cur != prev) {
        set_h_in_front_near_time(i, 0.0);
        set_h_in_front_near_lane_time(i, 0.0);
        return;
    }
    uint f = horse_by_order(cur - 1);
    float f_dist = h_distance(f), f_lane = h_lane_distance(f);
    if (f_dist - h_distance(i) > p_infront_near_distance()) {
        set_h_in_front_near_time(i, 0.0);
        set_h_in_front_near_lane_time(i, 0.0);
    } else {
        set_h_in_front_near_time(i, h_in_front_near_time(i) + dt);
        if (abs(h_lane_distance(i) - f_lane) <= p_infront_near_lane_distance()) set_h_in_front_near_lane_time(i, h_in_front_near_lane_time(i) + dt);
        else set_h_in_front_near_lane_time(i, 0.0);
    }
}

void update_behind_horse_near_time(uint i, float dt) {
    int cur = h_cur_order(i), prev = h_prev_order(i);
    if (cur != r_last_order() && cur == prev) {
        uint b = horse_by_order(cur + 1);
        float b_dist = h_distance(b), b_lane = h_lane_distance(b);
        if (h_distance(i) - b_dist <= p_behind_near_distance()) {
            set_h_behind_near_time(i, h_behind_near_time(i) + dt);
            if (abs(h_lane_distance(i) - b_lane) <= p_behind_near_lane_distance()) set_h_behind_near_lane_time(i, h_behind_near_lane_time(i) + dt);
            else set_h_behind_near_lane_time(i, 0.0);
            return;
        }
    }
    set_h_behind_near_time(i, 0.0);
    set_h_behind_near_lane_time(i, 0.0);
}

void update_behind_horse_near_time_param_set(uint i, float dt) {
    int cur = h_cur_order(i), prev = h_prev_order(i);
    int np = p_behind_near_param_count();
    if (cur != r_last_order() && cur == prev) {
        uint b = horse_by_order(cur + 1);
        float d_gap = h_distance(i) - h_distance(b);
        float l_gap = abs(h_lane_distance(i) - h_lane_distance(b));
        for (int k = 0; k < np; k++) {
            if (p_behind_near_param_distance(uint(k)) < d_gap || p_behind_near_param_lane_distance(uint(k)) < l_gap) set_h_behind_near_lane_times(i, uint(k), 0.0);
            else set_h_behind_near_lane_times(i, uint(k), h_behind_near_lane_times(i, uint(k)) + dt);
        }
    } else {
        for (int k = 0; k < np; k++) set_h_behind_near_lane_times(i, uint(k), 0.0);
    }
}

void post_update(float dt) {
    for (int i = 0; i < r_num_horses(); i++) {
        update_in_front_horse_near_time(uint(i), dt);
        update_behind_horse_near_time(uint(i), dt);
        update_behind_horse_near_time_param_set(uint(i), dt);
    }
}

#endif
