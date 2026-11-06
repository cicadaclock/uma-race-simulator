// Stage::Field: field::update_overlap, compete_top::update (spot struggle), compete_fight::update (dueling) and
// pacemaker::update (from the July 2026 courses on). Events (record::add_horse_event) are output only and not
// simulated here. Needs exact.glsl, race_state.glsl, sort.glsl, modifiers.glsl, stats.glsl and transcend.glsl.

#ifndef UMA_FIELD_GLSL
#define UMA_FIELD_GLSL

#include "queries.glsl"

// field::update_overlap: in (truncated) lane order, a horse closer than 0.4 m and 1/45 lane to the previous one is
// pushed out to 1/45 lane past it. NaN counts as overlap.
void update_overlap() {
    int n = r_num_horses();
    for (int k = 0; k < n; k++) sort_h[k] = k;
    sort_mode = SORT_LANE;
    if (!intro_sort(n)) set_r_gpu_fault(r_gpu_fault() | 16u);
    set_force_check_next_lane_one_time(uint(sort_h[0]), false);
    float gap = uintBitsToFloat(0x3cb60b61u);  // 0.022222223
    for (int k = 1; k < n; k++) {
        uint prev = uint(sort_h[k - 1]), cur = uint(sort_h[k]);
        float d = abs(h_distance(cur) - h_distance(prev));
        float l = abs(h_lane_distance(cur) - h_lane_distance(prev));
        if (!(d >= uintBitsToFloat(0x3ecccccdu)) && !(l >= gap)) {
            set_h_lane_distance(cur, h_lane_distance(prev) + gap);
            set_force_check_next_lane_one_time(cur, true);
        }
    }
}

// ---- compete_top.rs

bool ct_in_group(uint i, int ex) {
    return h_running_style(i) == STYLE_NIGE && h_running_style_ex(i) == ex && !h_is_finished(i);
}

void compete_top_start_groups() {
    int n = r_num_horses();
    for (int pass = 0; pass < 2; pass++) {
        int k = pass == 0 ? 1 : 0;  // Oonige group first, then Nige
        int ex = k == 1 ? STYLE_EX_OONIGE : STYLE_EX_NONE;
        int limit = k == 0 ? p_ct_nige_count() : p_ct_oonige_count();
        if (r_compete_top_start_count(uint(k)) >= limit) continue;
        int top = -1;
        for (int m = 0; m < r_style_len(uint(STYLE_NIGE)); m++) {
            int j = r_style_horse(uint(STYLE_NIGE) * MAX_HORSES + uint(m));
            if (ct_in_group(uint(j), ex)) {
                top = j;
                break;
            }
        }
        if (top < 0) continue;
        int to = h_cur_order(uint(top));
        bool ahead = false;
        for (int j = 0; j < n; j++) {
            if (h_cur_order(uint(j)) < to && (h_running_style(uint(j)) > STYLE_NIGE || (k == 1 && h_running_style_ex(uint(j)) != STYLE_EX_OONIGE))) ahead = true;
        }
        if (ahead) continue;
        float td = h_distance(uint(top)), tl = h_lane_distance(uint(top));
        int members[MAX_HORSES];
        int count = 1;
        members[0] = top;
        for (int j = 0; j < n; j++) {
            if (j != top && ct_in_group(uint(j), ex) && abs(h_distance(uint(j)) - td) < p_ct_distance_gap1() && abs(h_lane_distance(uint(j)) - tl) < p_ct_lane_gap1()) {
                members[count] = j;
                count++;
            }
        }
        bool early = false;
        for (int m = 0; m < count; m++) early = early || calc_section(uint(members[m])) <= p_ct_check_end_section();
        if (count < 2 || !early) continue;
        set_r_compete_top_start_count(uint(k), r_compete_top_start_count(uint(k)) + 1);
        for (int m = 0; m < count; m++) {
            uint i = uint(members[m]);
            float guts = stat_guts(i);
            set_h_is_compete_top(i, true);
            set_h_compete_top_count(i, h_compete_top_count(i) + 1);
            set_h_compete_top_group(i, k);
            set_h_compete_top_remain_time(i, mathf_pow(p_ct_time_coef1() * guts, p_ct_time_coef2()) * p_ct_time_coef3() * h_proper_running_style_coef(i));
            // conserve_power::on_compete_top_start
            set_h_conserved_power(i, h_conserved_power(i) * p_cp_decrease_coef_compete_top());
            set_h_conserve_activity_coef(i, h_conserve_activity_coef(i) * p_cp_decrease_accel_coef_compete_top());
        }
    }
}

bool ct_near(uint i) {
    for (uint j = 0u; j < uint(r_num_horses()); j++) {
        if (j != i && h_compete_top_group(j) == h_compete_top_group(i) && abs(h_distance(j) - h_distance(i)) <= p_ct_distance_gap2() && abs(h_lane_distance(j) - h_lane_distance(i)) <= p_ct_lane_gap2()) return true;
    }
    return false;
}

void compete_top_update(float dt) {
    int n = r_num_horses();
    bool started = false;
    for (int i = 0; i < n; i++) started = started || h_distance(uint(i)) >= p_ct_check_start_distance();
    if (started) compete_top_start_groups();
    // Groups whose spot struggle ends this frame, before any flag changes.
    int ended[MAX_HORSES];
    int ended_count = 0;
    for (int i = 0; i < n; i++) {
        if (h_is_compete_top(uint(i)) && calc_section(uint(i)) >= p_ct_end_section()) {
            ended[ended_count] = h_compete_top_group(uint(i));
            ended_count++;
        }
    }
    for (int i = 0; i < n; i++) {
        uint h = uint(i);
        if (!h_is_compete_top(h)) continue;
        if (!ct_near(h)) {
            set_h_is_compete_top(h, false);
            continue;
        }
        float remain = h_compete_top_remain_time(h) - dt;
        set_h_compete_top_remain_time(h, remain);
        bool group_ended = false;
        for (int e = 0; e < ended_count; e++) group_ended = group_ended || ended[e] == h_compete_top_group(h);
        if (remain <= 0.0 || group_ended) set_h_is_compete_top(h, false);
    }
}

// ---- compete_fight.rs

bool cf_hp_ok(uint h, int per) { return h_hp(h) >= h_max_hp(h) * float(per) * uintBitsToFloat(0x3c23d70au); }

void compete_fight_start(uint i, int group) {
    set_h_is_compete_fight(i, true);
    set_h_compete_fight_count(i, h_compete_fight_count(i) + 1);
    set_h_compete_fight_group(i, group);
}

void compete_fight_update() {
    int n = r_num_horses();
    float top = float(n) * float(p_cf_target_order_per()) * uintBitsToFloat(0x3c23d70au);
    for (int ii = 0; ii < n; ii++) {
        uint i = uint(ii);
        if (h_is_compete_fight(i) || h_compete_fight_count(i) > 0) continue;
        int partner = -1;
        for (int m = 0; m < h_cf_near_len(i); m++) {
            uint o = uint(h_cf_near(i, uint(m)));
            if (h_ar_cf_time(i, o) >= p_cf_target_continue_time() && h_ar_speed_gap_abs(i, o) < p_cf_speed_gap() && cf_hp_ok(i, p_cf_hp_per2()) && cf_hp_ok(o, p_cf_hp_per2())
                && (float(h_cur_order(i)) < top || float(h_cur_order(o)) < top)) {
                partner = int(o);
                break;
            }
        }
        if (partner < 0) continue;
        uint o = uint(partner);
        if (!h_is_compete_fight(o) && h_compete_fight_count(o) > 0) continue;
        int g;
        if (h_is_compete_fight(o)) g = h_compete_fight_group(o);
        else {
            int mx = -1;
            for (int j = 0; j < n; j++) mx = max(mx, h_compete_fight_group(uint(j)));
            g = mx + 1;
        }
        compete_fight_start(i, g);
        if (!h_is_compete_fight(o)) compete_fight_start(o, g);
    }
    for (int ii = 0; ii < n; ii++) {
        uint i = uint(ii);
        if (!h_is_compete_fight(i)) continue;
        bool near = false;
        for (int j = 0; j < n; j++) {
            if (uint(j) != i && h_compete_fight_group(uint(j)) == h_compete_fight_group(i) && abs(h_distance(uint(j)) - h_distance(i)) <= p_cf_target_continue_distance()) near = true;
        }
        if (!near || h_hp(i) < h_max_hp(i) * float(p_cf_hp_per()) * uintBitsToFloat(0x3c23d70au)) set_h_is_compete_fight(i, false);
    }
}

// ---- pacemaker.rs (the July 2026 rules; the earlier courses set the candidate in EndOfFrame)

int pacemaker_successor(uint leader) {
    int style = h_running_style(leader);
    if (style < STYLE_SASHI) return int(leader);
    float range = p_pk_pacemaker_change_forward_range();
    int best = -1;
    for (int j = 0; j < r_num_horses(); j++) {
        if (h_running_style(uint(j)) < style && h_distance(leader) - h_distance(uint(j)) <= range) {
            if (best < 0 || h_cur_order(uint(j)) < h_cur_order(uint(best))) best = j;
        }
    }
    return best < 0 ? int(leader) : best;
}

void pacemaker_update() {
    int most = r_most_forward_running_style();
    int cand = r_style_top_horse(uint(most));
    if (cand < 0) return;
    int lead = r_num_horses() > 0 ? r_order_horse(0u) : -1;
    bool not_most = r_pm_horse() >= 0 && lead >= 0 && h_running_style(uint(lead)) != most;
    int sec = calc_section(uint(lead >= 0 ? lead : cand));
    int heir = lead >= 0 ? pacemaker_successor(uint(lead)) : -1;
    float threshold = p_pk_pacemaker_change_not_most_forward_count();
    int end_section = p_pk_end_section();
    bool repick = r_pm_horse() < 0 || (sec > r_pm_section() && sec <= end_section);
    if (r_pm_overriding() && !not_most && !repick) {
        set_r_pm_overriding(false);
        set_r_pm_not_most_forward_count(0);
        set_r_pm_section_pick(cand);
        set_r_pm_horse(cand);
    }
    if (repick) {
        set_r_pm_section(sec);
        if (not_most) {
            set_r_pm_not_most_forward_count(r_pm_not_most_forward_count() + 1);
            if (float(r_pm_not_most_forward_count()) >= threshold && heir != r_pm_horse()) {
                set_r_pm_not_most_forward_count(0);
                set_r_pm_overriding(true);
                set_r_pm_horse(heir);
            }
        } else if (!r_pm_overriding()) {
            set_r_pm_section_pick(cand);
            set_r_pm_horse(cand);
        }
    }
}

void field_stage(float dt) {
    update_overlap();
    compete_top_update(dt);
    compete_fight_update();
    pacemaker_update();
}

#endif
