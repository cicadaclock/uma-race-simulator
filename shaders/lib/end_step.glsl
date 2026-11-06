// tick::end_step without the recording: the distance spread (race::update_distance_diff_max), then the course
// events (course::event_manager_update). The event tables are in the race's course block (course_buffer.glsl).
// Needs exact.glsl, race_state.glsl, queries.glsl and course_buffer.glsl.

#ifndef UMA_END_STEP_GLSL
#define UMA_END_STEP_GLSL

const int EV_CORNER = 0, EV_GROUND_CHANGE = 1, EV_STRAIGHT = 2, EV_LANE_MAX_CHANGE = 3, EV_SLOPE = 11;
const uint FAULT_END_STEP = 512u;

// The largest spread of the unfinished horses so far.
void update_distance_diff_max() {
    float mn = uintBitsToFloat(0x7f7fffffu), mx = -uintBitsToFloat(0x7f7fffffu);
    for (uint h = 0u; h < uint(r_num_horses()); h++) {
        if (h_is_finished(h)) continue;
        float d = h_distance(h);
        if (mn > d) mn = d;
        if (d > mx) mx = d;
    }
    float d = mx - mn;
    if (d > r_distance_diff_max()) set_r_distance_diff_max(d);
}

uint cb() { return r_course_base(); }
// The k-th record (4 words) of the table whose offset and count are at header word `hdr`.
uint course_rec(uint hdr, uint k) { return cb() + crs[cb() + hdr] + 4u * k; }
uint course_count(uint hdr) { return crs[cb() + hdr + 1u]; }

// Event value k (Rust indexes them: out of range panics).
int ev_value(uint ev, uint k) {
    if (k >= crs[ev + 2u]) {
        gpu_fault(FAULT_END_STEP);
        return 0;
    }
    return int(crs[cb() + crs[ev + 3u] + k]);
}

void event_straight_change(uint i, uint ev) {
    uint n = crs[ev + 2u];
    if (n == 0u) return;
    bool is_start = ev_value(ev, 0u) == 1;
    if (is_start && n < 2u) return;
    int ft = is_start ? ev_value(ev, 1u) : 0;
    float dist = h_distance(i);
    int found = -1;
    for (uint k = 0u; k < course_count(18u); k++) {
        uint s = course_rec(18u, k);
        if (int(crs[s]) == ft && !(crs_f(s + 1u) > dist) && !(dist > crs_f(s + 2u))) {
            found = int(crs[s + 3u]);
            break;
        }
    }
    bool was_straight = h_is_straight(i);
    set_h_is_straight(i, is_start);
    set_h_cur_straight_front_type(i, ft);
    set_h_cur_straight_type_number(i, -1);
    if (is_start) {
        if (found != -1) set_h_cur_straight_type_number(i, found);
        set_h_cur_corner(i, 0);
        // Entering the last straight (after the final corner) records the order, once.
        bool is_straight_last = h_is_straight(i) && h_is_final_corner(i);
        if (!was_straight && is_straight_last && h_last_straight_order(i) == -1) set_h_last_straight_order(i, h_cur_order(i));
    }
}

void event_corner_change(uint i, uint ev) {
    if (crs[ev + 2u] < 2u) return;
    int no = ev_value(ev, 0u);
    float dist = h_distance(i);
    int c = -1;
    for (uint k = 0u; k < course_count(20u); k++) {
        uint r = course_rec(20u, k);
        if (int(crs[r]) == no && crs_f(r + 1u) <= dist && dist <= crs_f(r + 2u)) {
            c = int(r);
            break;
        }
    }
    set_h_is_final_corner(i, c >= 0 && crs[uint(c) + 3u] != 0u);
    set_h_is_straight(i, false);
    set_h_cur_corner(i, no);
    set_h_corner_end_distance(i, c >= 0 ? crs_f(uint(c) + 2u) : dist + 200.0);
}

void event_slope_change(uint i, uint ev) {
    if (crs[ev + 2u] != 2u) return;
    float per = div_rn32(float(ev_value(ev, 0u)), 10000.0);
    int ty = per > 0.0 ? 1 : (per < 0.0 ? 2 : 0);
    float dist = h_distance(i);
    for (uint k = 0u; k < course_count(22u); k++) {
        uint r = course_rec(22u, k);
        if (int(crs[r]) == ty && crs_f(r + 1u) <= dist && dist <= crs_f(r + 2u)) {
            set_h_slope_type(i, ty);
            set_h_slope_per(i, per);
            set_h_slope_end_distance(i, crs_f(r + 2u));
            return;
        }
    }
    set_h_slope_type(i, 0);
    set_h_slope_per(i, 0.0);
}

// exec_event: once the horse reaches it, every handler (each filters on the param type).
void exec_event(uint event_no, uint i, float dist) {
    uint ev = course_rec(16u, event_no);
    if (!(h_last_event(i) < int(event_no) && dist >= crs_f(ev))) return;
    set_h_last_event(i, int(event_no));
    int ty = int(crs[ev + 1u]);
    if (ty == EV_GROUND_CHANGE) {
        int g = ev_value(ev, 0u);
        if (h_ground_type(i) != g) set_h_ground_type(i, g);
    } else if (ty == EV_STRAIGHT) {
        event_straight_change(i, ev);
    } else if (ty == EV_CORNER) {
        event_corner_change(i, ev);
    } else if (ty == EV_LANE_MAX_CHANGE) {
        if (!r_is_story()) set_h_lane_distance_max(i, div_rn32(float(ev_value(ev, 0u)), 10000.0));
    } else if (ty == EV_SLOPE) {
        event_slope_change(i, ev);
    }
}

// Events in list order (sorted by distance), horses in index order. As on the CPU, events below every horse's last
// one and past the farthest horse are skipped (no horse could execute them).
void event_manager_update() {
    uint n = uint(r_num_horses());
    int first = 0;
    if (n > 0u) {
        int m = h_last_event(0u);
        for (uint h = 1u; h < n; h++) {
            if (h_last_event(h) < m) m = h_last_event(h);
        }
        first = m + 1 > 0 ? m + 1 : 0;
    }
    float max_dist = -uintBitsToFloat(0x7f7fffffu);
    for (uint h = 0u; h < n; h++) {
        if (h_distance(h) > max_dist) max_dist = h_distance(h);
    }
    for (uint e = uint(first); e < course_count(16u); e++) {
        if (crs_f(course_rec(16u, e)) > max_dist) break;
        for (uint h = 0u; h < n; h++) exec_event(e, h, h_distance(h));
    }
}

void end_step(bool is_end) {
    if (!is_end) update_distance_diff_max();
    event_manager_update();
}

#endif
