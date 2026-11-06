// sim/skill (mod.rs, trigger/*, ability.rs, value.rs, target.rs) on the GPU: the per-frame skill update of
// Stage::SkillsFirst / SkillsSecond. The program and the state (skill_state.glsl) are built by gpu::skills.
//
// The CPU skips skills and conditions that are certainly idle (the skill walk, `is_dormant`, a condition's wake
// thresholds). Updating them anyway changes nothing but the `is_activated` of pure terms, which nothing reads and
// which neither side keeps, so this follows the plain update. Not supported: observed and muted skills (the program
// flags them and the kernel faults). Skill events (the recording) are not produced.
//
// Needs exact.glsl, race_state.glsl, net_random.glsl, sort.glsl, modifiers.glsl, queries.glsl, stats.glsl,
// skill_layout.glsl and skill_state.glsl.

#ifndef UMA_SKILLS_GLSL
#define UMA_SKILLS_GLSL

// ---- small helpers

bool check_value(int lh, uint op, int rh) {
    if (op == 0u) return lh == rh;
    if (op == 1u) return lh != rh;
    if (op == 2u) return lh > rh;
    if (op == 3u) return lh >= rh;
    if (op == 4u) return lh < rh;
    return lh <= rh;
}
bool check_value_float(float lh, uint op, int rh) {
    float r = float(rh);
    if (op == 0u) return mathf_approximately(lh, r);
    if (op == 1u) return !mathf_approximately(lh, r);
    if (op == 2u) return r < lh;
    if (op == 3u) return r <= lh;
    if (op == 4u) return lh < r;
    return lh <= r;
}
bool check_value_input_float(float lh, uint op, float r) {
    if (op == 0u) return mathf_approximately(lh, r);
    if (op == 1u) return !mathf_approximately(lh, r);
    if (op == 2u) return lh > r;
    if (op == 3u) return lh >= r;
    if (op == 4u) return lh < r;
    return lh <= r;
}

bool is_skill_demerit(int t, float v) {
    if ((t >= 1 && t <= 5) || t == 8 || t == 9 || t == 21 || t == 22 || t == 27 || t == 28 || (t >= 30 && t <= 32)) return v < 0.0;
    if (t == 10) return v > 1.0;
    if (t == 13 || t == 14) return true;
    if (t == 29) return !(v < 0.0);
    return false;
}

int activate_skill_count(uint h) {
    return h_act_count_by_phase(h, 0u) + h_act_count_by_phase(h, 1u) + h_act_count_by_phase(h, 2u) + h_act_count_by_phase(h, 3u);
}

// field::first_horse / horse_by_order / last_horse_index
uint first_horse() { return uint(r_order_horse(0u)); }
uint horse_by_order(int order) {
    if (order < 0 || order >= r_num_horses()) {
        gpu_fault(FAULT_SKILLS);
        return 0u;
    }
    return uint(r_order_horse(uint(order)));
}
int last_horse_index() { return horse_index_at_order(r_num_horses() - 1); }

// asc_lookup: the first entry with x < threshold, else 1.
float asc_lookup(uint table, uint count, float x) {
    for (uint k = 0u; k < count; k++) {
        if (x < Pf(table + 2u * k)) return Pf(table + 2u * k + 1u);
    }
    return 1.0;
}

// ---- value_lh

int lane_type(float lane) {
    if (lane <= uintBitsToFloat(0x3e4ccccdu)) return 0;                                   // 0.2f
    if (uintBitsToFloat(0x3ecccccdu) < lane) return (uintBitsToFloat(0x3f19999au) < lane ? 1 : 0) + 2;  // 0.4f, 0.6f
    return 1;
}

// One exit (a result variable): lavapipe returned the wrong case from this dispatch with a return per case.
int value_lh(uint owner, uint kind, uint pt) {
    uint h = owner;
    int cd = r_course_distance();
    int r = 0;
    switch (kind) {
        case K_LATCHED: { r = Pi(pt + PT_P0); break; }
        case K_COURSE_CORNER_COUNT: { r = Pi(PR_CORNER_COUNT); break; }
        case K_GRADE: { r = Pi(PR_GRADE); break; }
        case K_RACE_ID: { r = Pi(PR_RACE_ID); break; }
        case K_TIME: { r = Pi(PR_TIME); break; }
        case K_COURSE_DISTANCE: { r = cd; break; }
        case K_IS_DIRT_GRADE: { r = Pi(PR_IS_DIRT_GRADE); break; }
        case K_RACE_ACCUMULATE_TIME: { r = floor_to_int(r_accumulate_time()); break; }
        case K_SLOPE: { r = h_slope_type(h); break; }
        case K_CORNER: { r = h_cur_corner(h); break; }
        case K_FURLONG: { r = floor_to_int((h_distance(h) + 0.5) * uintBitsToFloat(0x3ba3d70au)); break; }  // 0.005f
        case K_ORDER: { r = h_cur_order(h) + 1; break; }
        case K_ORDER_LAST_SPURT: { r = h_last_spurt_order(h) + 1; break; }
        case K_ORDER_FINAL_CORNER_END: { r = h_final_corner_end_order(h) + 1; break; }
        case K_GATE_NUMBER: { r = int(h) + 1; break; }
        case K_POST_NUMBER: { r = Pi(ph(h) + PH_POST_NUMBER); break; }
        case K_HORSE_DISTANCE_RATE: { r = f32_to_i32(h_distance(h)); break; }
        case K_HORSE_REMAIN_DISTANCE: { r = cd - floor_to_int(h_distance(h)); break; }
        case K_HORSE_DISTANCE_DIFF_RATE: {
            float first = h_distance(first_horse());
            float last = h_distance(uint(last_horse_index()));
            float span = first - last;
            float num = first - h_distance(h);
            { r = mathf_approximately(span, 0.0) ? 0 : floor_to_int(div_rn32(num, span) * 100.0); break; }
        }
        case K_OTHER_HORSE_BEHIND: { r = r_num_race_horses() - h_cur_order(h) - 1; break; }
        case K_BE_BLOCKED: { r = h_block_horse(h) >= 0 && !is_enable_in_move(h) && !is_enable_out_move(h) ? 1 : 0; break; }
        case K_BE_BLOCKED_FRONT: { r = h_block_horse(h) >= 0 ? 1 : 0; break; }
        case K_BE_BLOCKED_SIDE: { r = h_block_horse_in(h) >= 0 || h_block_horse_out(h) >= 0 ? 1 : 0; break; }
        case K_BE_BLOCKED_ALL_CONTINUE_TIME: {
            float f = h_block_front_continue_time(h), s = h_block_side_continue_time(h);
            { r = f32_to_i32(s <= f ? s : f); break; }
        }
        case K_BE_BLOCKED_FRONT_CONTINUE_TIME: { r = f32_to_i32(h_block_front_continue_time(h)); break; }
        case K_BE_BLOCKED_SIDE_CONTINUE_TIME: { r = f32_to_i32(h_block_side_continue_time(h)); break; }
        case K_BE_BLOCKED_FRONT_HP_EMPTY: { r = h_block_horse(h) >= 0 && h_hp(uint(h_block_horse(h))) <= 0.0 ? 1 : 0; break; }
        case K_BE_BLOCKED_FRONT_HP_EMPTY_CONTINUE_TIME: { r = f32_to_i32(h_block_front_hp_empty_time(h)); break; }
        case K_TEMPTATION_COUNT: { r = h_temptation_count(h); break; }
        case K_COMPETE_FIGHT_COUNT: { r = h_compete_fight_count(h); break; }
        case K_GOOD_START: { r = h_is_good_start(h) ? 1 : 0; break; }
        case K_BAD_START: { r = h_is_bad_start(h) ? 1 : 0; break; }
        case K_BASE_SPEED: { r = f32_to_i32(Pf(ph(h) + PH_BASE_SPEED)); break; }
        case K_BASE_STAMINA: { r = f32_to_i32(Pf(ph(h) + PH_BASE_STAMINA)); break; }
        case K_BASE_POWER: { r = f32_to_i32(Pf(ph(h) + PH_BASE_POW)); break; }
        case K_BASE_GUTS: { r = f32_to_i32(Pf(ph(h) + PH_BASE_GUTS)); break; }
        case K_BASE_WIZ: { r = f32_to_i32(Pf(ph(h) + PH_BASE_WIZ_SKILLS)); break; }
        case K_HP_PER: { r = f32_to_i32(div_rn32(h_hp(h), h_max_hp(h)) * 100.0); break; }
        case K_HP_EMPTY: { r = h_hp(h) <= 0.0 ? 1 : 0; break; }
        case K_CURRENT_SPEED: { r = f32_to_i32(h_last_speed(h)); break; }
        case K_SPEED_ORDER: {
            float own = stat_speed(h);
            int c = 1;
            for (uint o = 0u; o < uint(r_num_horses()); o++) {
                if (stat_speed(o) > own) c++;
            }
            { r = c; break; }
        }
        case K_VISIBLE_HORSE: { r = h_visible_horse_count(h); break; }
        case K_CHANGE_ORDER: case K_CHANGE_ORDER_ONE_TIME: { r = h_cur_order(h) - h_prev_order(h); break; }
        case K_LANE_TYPE: { r = lane_type(h_lane_distance(h)); break; }
        case K_LANE_PER: { r = floor_to_int(h_lane_distance_per_in_all_horses(h)); break; }
        case K_MOVE_LANE: { r = h_move_lane_continue_time(h) < 1.0 ? 0 : (h_move_lane_dir(h) != 0 ? 2 : 1); break; }
        case K_OVERTAKE: { r = is_overtake(h) ? 1 : 0; break; }
        case K_OVERTAKE_TARGET_CONTINUE_TIME: { r = floor_to_int(h_overtake_target_time(h)); break; }
        case K_OVERTAKE_TARGET_NO_ORDER_UP_TIME: { r = floor_to_int(h_no_order_up_time(h)); break; }
        case K_OVERTAKE_TARGET_NO_ORDER_DOWN_TIME: { r = floor_to_int(h_no_order_down_time(h)); break; }
        case K_SURROUNDED: { r = h_is_surround_out(h) && h_is_surround_ahead(h) && h_is_surround_behind(h) ? 1 : 0; break; }
        case K_NEAR_HORSE_COUNT: { r = h_near_horse_count(h); break; }
        case K_NEAR_AROUND_ACCUMULATE_TIME: { r = floor_to_int(h_near_around_time(h)); break; }
        case K_DISTANCE_DIFF_TOP: { r = floor_to_int(h_distance(first_horse()) - h_distance(h)); break; }
        case K_DISTANCE_DIFF_INFRONT: { r = floor_to_int(h_distance(horse_by_order(h_cur_order(h) - 1)) - h_distance(h)); break; }
        case K_DISTANCE_DIFF_BEHIND: { r = floor_to_int(h_distance(h) - h_distance(horse_by_order(h_cur_order(h) + 1))); break; }
        case K_NEAR_FRONT_HORSE_COUNT: case K_NEAR_BEHIND_HORSE_COUNT: {
            bool front = kind == K_NEAR_FRONT_HORSE_COUNT;
            int c = 0;
            for (int k = 0; k < h_near_horse_count(h); k++) {
                float g = h_ar_distance_gap(h, uint(h_near_horses(h, uint(k))));
                if (front) {
                    if (g > 0.0) c++;
                } else if (!(g > 0.0 || g == 0.0)) {
                    c++;
                }
            }
            { r = c; break; }
        }
        case K_HORSE_BEHIND_IN: { r = h_lane_distance(horse_by_order(h_cur_order(h) + 1)) < h_lane_distance(h) ? 1 : 0; break; }
        case K_HORSE_BEHIND_OUT: { r = h_lane_distance(h) < h_lane_distance(horse_by_order(h_cur_order(h) + 1)) ? 1 : 0; break; }
        case K_HORSE_INFRONT_NEAR_TIME: { r = floor_to_int(h_in_front_near_time(h)); break; }
        case K_HORSE_BEHIND_NEAR_TIME: { r = floor_to_int(h_behind_near_time(h)); break; }
        case K_HORSE_INFRONT_NEAR_LANE_TIME: { r = floor_to_int(h_in_front_near_lane_time(h)); break; }
        case K_HORSE_BEHIND_NEAR_LANE_TIME: { r = floor_to_int(h_behind_near_lane_time(h)); break; }
        case K_HORSE_BEHIND_NEAR_LANE_TIME_PARAM_SET: {
            uint p = P(pt + PT_P0);
            { r = floor_to_int(p < uint(p_behind_near_param_count()) ? h_behind_near_lane_times(h, p) : 0.0); break; }
        }
        case K_ACTIVATE_HEAL_SKILL_COUNT: { r = h_act_heal_count(h); break; }
        case K_ACTIVATE_COUNT: { r = activate_skill_count(h); break; }
        case K_ACTIVATE_COUNT_PHASE: {
            uint m = P(pt + PT_P0);
            int c = 0;
            for (uint k = 0u; k < 4u; k++) {
                if ((m & (1u << k)) != 0u) c += h_act_count_by_phase(h, k);
            }
            { r = c; break; }
        }
        case K_ACTIVATE_COUNT_LATER_HALF: { r = h_act_count_by_distance(h, 1u); break; }
        case K_ACTIVATE_COUNT_ALL_TEAM: {
            int t = team_id(h), c = 0;
            for (uint j = 0u; j < uint(r_num_horses()); j++) {
                int tj = team_id(j);
                if (tj != 0 && t != 0 && tj == t) c += activate_skill_count(j);
            }
            { r = c; break; }
        }
        case K_CHANGE_ORDER_UP_PHASE_MIDDLE: { r = h_ocu_phase_middle(h); break; }
        case K_CHANGE_ORDER_UP_PHASE_END_AFTER: { r = h_ocu_phase_end_after(h); break; }
        case K_CHANGE_ORDER_UP_CORNER: { r = h_ocu_corner(h); break; }
        case K_CHANGE_ORDER_UP_LAST_SPURT: { r = h_ocu_last_spurt(h); break; }
        case K_CHANGE_ORDER_UP_FINAL_CORNER_AFTER: { r = h_ocu_final_corner_after(h); break; }
        case K_CHANGE_ORDER_DOWN_PHASE_MIDDLE: { r = h_ocd_phase_middle(h); break; }
        case K_CHANGE_ORDER_DOWN_PHASE_END_AFTER: { r = h_ocd_phase_end_after(h); break; }
        case K_CHANGE_ORDER_DOWN_CORNER: { r = h_ocd_corner(h); break; }
        case K_CHANGE_ORDER_DOWN_LAST_SPURT: { r = h_ocd_last_spurt(h); break; }
        case K_CHANGE_ORDER_DOWN_FINAL_CORNER_AFTER: { r = h_ocd_final_corner_after(h); break; }
        case K_GROUND_TYPE: { r = Pi(PR_GROUND); break; }
        case K_GROUND_CONDITION: { r = Pi(PR_GROUND_CONDITION); break; }
        case K_COURSE_ROTATION: { r = Pi(PR_TURN); break; }
        case K_COURSE_DISTANCE_TYPE: { r = r_course_distance_type(); break; }
        case K_WEATHER: { r = Pi(PR_WEATHER); break; }
        case K_SEASON: { r = Pi(PR_SEASON); break; }
        case K_RACE_TRACK_ID: { r = Pi(PR_TRACK_ID); break; }
        case K_STRAIGHT_FRONT_TYPE: { r = h_cur_straight_front_type(h); break; }
        case K_FINAL_CORNER: { r = h_is_final_corner(h) ? 1 : 0; break; }
        case K_RUNNING_STYLE: { r = h_running_style(h); break; }
        case K_MOTIVATION: { r = Pi(ph(h) + PH_MOTIVATION); break; }
        case K_POPULARITY: { r = Pi(ph(h) + PH_POPULARITY) + 1; break; }
        case K_PHASE: { r = h_phase(h); break; }
        case K_TEMPTATION: { r = h_is_temptation(h) ? 1 : 0; break; }
        case K_IS_LAST_SPURT: { r = is_last_spurt(h) ? 1 : 0; break; }
        case K_LAST_SPURT: {
            int c = h_ls_calc_result(h);
            { r = c == 1 ? 2 : (c == 2 ? 1 : 0); break; }
        }
        case K_RUNNING_STYLE_TEMPTATION_COUNT: {
            uint style = P(pt + PT_P0);
            bool only_opp = P(pt + PT_P1) != 0u;
            int n = r_style_rushed_len(style);
            if (!only_opp) { r = n; break; }
            int t = team_id(h), c = 0;
            for (int k = 0; k < n; k++) {
                uint j = uint(r_style_rushed_horse(style * MAX_HORSES + uint(k)));
                int tj = team_id(j);
                if (j != h && (t == 0 || tj == 0 || t != tj)) c++;
            }
            { r = c; break; }
        }
        case K_TEMPTATION_BEHIND: case K_TEMPTATION_INFRONT: {
            bool behind = kind == K_TEMPTATION_BEHIND;
            bool only_opp = P(pt + PT_P0) != 0u;
            int t = team_id(h), c = 0;
            for (int k = 0; k < r_rushed_len(); k++) {
                uint j = uint(r_rushed_horse(uint(k)));
                int tj = team_id(j);
                bool pass = !only_opp || t == 0 || tj == 0 || t != tj;
                if (pass && (behind ? h_cur_order(j) > h_cur_order(h) : h_cur_order(j) < h_cur_order(h))) c++;
            }
            { r = c; break; }
        }
        case K_HORSE_VIEWER_ID_REMAIN_DISTANCE: {
            uint m = P(pt + PT_P0);
            if (m == 0u) {
                gpu_fault(FAULT_SKILLS);  // the Rust panics (no horse with a viewer id)
                { r = 0; break; }
            }
            float mx = 0.0;
            bool any = false;
            for (uint j = 0u; j < MAX_HORSES; j++) {
                if ((m & (1u << j)) == 0u) continue;
                float d = h_distance(j);
                if (!any) mx = d;
                else if (d > mx) mx = d;
                any = true;
            }
            { r = cd - floor_to_int(mx); break; }
        }
        case K_SAME_SKILL_HORSE_COUNT: { r = 0; break; }  // read from its state by check_activate
    default:
        { r = 0; break; }
    }
    return r;
}

// ---- checks

// Whether skill id `id` is among horse j's skills.
bool has_skill(uint j, int id) {
    for (uint s = 0u; s < skill_count(j); s++) {
        if (skill_id(j, s) == id) return true;
    }
    return false;
}

// OtherCharacterActivateAdvantageSkill: a non-teammate's previous-frame detail has a buff of ability type t.
bool other_activate_advantage(uint owner, int target_type) {
    int b = team_id(owner);
    for (uint hi = 0u; hi < uint(r_num_horses()); hi++) {
        uint cached = P(ph(hi) + PH_CACHED_DETAILS);
        if (hi == owner || list_count(cached) == 0u) continue;
        int a = team_id(hi);
        if (a != 0 && b != 0 && a == b) continue;
        for (uint k = 0u; k < list_count(cached); k++) {
            uint code = list_get(cached, k);
            uint det = pd(hi, code / 2u, code % 2u);
            for (uint ab = 0u; ab < P(det + PD_ABILITY_COUNT); ab++) {
                uint pa = P(det + PD_ABILITIES) + ab * PA_WORDS;
                int ty = 0;
                float v = 0.0;
                if (P(pa + PA_PRESENT) != 0u) {
                    ty = Pi(pa + PA_TYPE);
                    v = Sf(P(pa + PA_STATE) + AA_VALUE_ON_ACTIVATE);
                }
                if (ty != target_type) continue;
                if (ty == 9) {
                    if (v > 0.0) return true;
                } else if (!is_skill_demerit(ty, v)) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Whether the owner's previous-frame details include a skill with one of the tags.
bool activate_tag_group(uint owner, uint tags, uint tag_count) {
    uint prev = P(ph(owner) + PH_PREV_DETAILS);
    for (uint k = 0u; k < list_count(prev); k++) {
        uint sp = ps(owner, list_get(prev, k) / 2u);
        for (uint t = 0u; t < P(sp + PS_TAG_COUNT); t++) {
            uint tag = P(P(sp + PS_TAGS) + t);
            for (uint g = 0u; g < tag_count; g++) {
                if (P(tags + g) == tag) return true;
            }
        }
    }
    return false;
}

// check_pure: terms without own state (and the stateless checks of the others). Every kind that compares
// value_lh shares the one call at the end (each call site is a full copy of value_lh once inlined); `gate` is the
// kind's precondition for reading it.
bool check_pure(uint owner, uint pt, uint st) {
    uint kind = P(pt + PT_KIND), op = P(pt + PT_OP);
    int rh = Pi(pt + PT_RH);
    uint h = owner;
    bool gate = true;
    switch (kind) {
        case K_NULL: return false;
        case K_ALWAYS: return true;
        case K_RANDOM_LOT: return Sb(st + 1u);
        case K_RACE_ACCUMULATE_TIME_RANDOM: return Si(st + 1u) == f32_to_i32(r_accumulate_time());
        case K_ORDER_RATE_IN_CONTINUE: return check_value(h_order_in_continue(h, P(pt + PT_P0)) ? 1 : 0, op, rh);
        case K_ORDER_RATE_OUT_CONTINUE: return check_value(h_order_out_continue(h, P(pt + PT_P0)) ? 1 : 0, op, rh);
        case K_ORDER_LAST_SPURT: gate = h_last_spurt_order(h) >= 0; break;
        case K_ORDER_FINAL_CORNER_END: gate = h_final_corner_end_order(h) != -1; break;
        case K_HP_PER: case K_HP_EMPTY: gate = h_is_param_initialized(h); break;
        case K_HP: return h_is_param_initialized(h) && check_value_float(h_hp(h), op, rh);
        case K_DISTANCE_DIFF_INFRONT: gate = h_cur_order(h) != 0; break;
        case K_DISTANCE_DIFF_BEHIND: case K_HORSE_BEHIND_IN: case K_HORSE_BEHIND_OUT: gate = int(h) != last_horse_index(); break;
        case K_BASHIN_DIFF_INFRONT: {
            if (h_cur_order(h) == 0) return false;
            float d = h_distance(horse_by_order(h_cur_order(h) - 1)) - h_distance(h);
            return check_value_float(d >= 0.0 ? div_rn32(d, 2.5) : 0.0, op, rh);
        }
        case K_BASHIN_DIFF_BEHIND: {
            if (int(h) == last_horse_index()) return false;
            float d = h_distance(h) - h_distance(horse_by_order(h_cur_order(h) + 1));
            return check_value_float(d >= 0.0 ? div_rn32(d, 2.5) : 0.0, op, rh);
        }
        case K_DISTANCE_DIFF_TOP_FLOAT: {
            float d = h_distance(first_horse()) - h_distance(h);
            return check_value_input_float(div_rn32(floor(d * 10.0), 10.0), op, Pf(pt + PT_P0));
        }
        case K_LAST_SPURT: gate = h_ls_calc_count(h) > 0; break;
        case K_EXIST_CHARA_ID: {
            bool e = P(pt + PT_P0) != 0u;
            return op == 0u ? e : (op == 1u ? !e : false);
        }
        case K_USED_PARTICULAR_SKILL_ID: return list_contains(P(ph(h) + PH_USED_IDS), P(pt + PT_P0));
        case K_OTHER_CHARACTER_ACTIVATE_ADVANTAGE_SKILL: return other_activate_advantage(h, Pi(pt + PT_P0));
        case K_ACTIVATE_SKILL_TAG_GROUP: return activate_tag_group(h, P(pt + PT_P0), P(pt + PT_P1));
        case K_FINAL_CORNER_LATER_HALF: {
            if (!(Sb(st + 1u) && Sb(st + 2u))) return false;
            float d = h_distance(h);
            return check_value(Sf(st + 3u) <= d && d <= Sf(st + 4u) ? 1 : 0, op, rh);
        }
        case K_DISTANCE: {
            float d = h_distance(h);
            return Sb(st + 3u) && Sb(st + 4u) && Sf(st + 1u) <= d && d <= Sf(st + 2u);
        }
        case K_ALL_CORNER_RANDOM: {
            uint n = S(st + 1u);
            if (!Sb(st + 2u) || n == 0u) return false;
            float d = h_distance(h);
            for (uint k = 0u; k < n; k++) {
                if (d < Sf(st + 3u + 2u * k)) return false;
                if (d <= Sf(st + 4u + 2u * k)) return true;
            }
            return false;
        }
        case K_SAME_SKILL_HORSE_COUNT: return check_value(Si(st + 1u), op, rh);
        default: break;
    }
    return gate && check_value(value_lh(h, kind, pt), op, rh);
}

// is_pure: terms whose update has no effect but is_activated.
bool term_is_pure(uint pt, uint st) {
    uint kind = P(pt + PT_KIND);
    if (kind == K_RANDOM_LOT_OTHER_ACTIVATE || kind == K_HP_EMPTY_ONE_TIME || kind == K_CHANGE_ORDER || kind == K_SAME_SKILL_HORSE_COUNT
        || kind == K_ACTIVATE_OTHER_SKILL_DETAIL || kind == K_IS_ACTIVATE_ANY_SKILL || kind == K_LAST_STRAIGHT_ONE_TIME) {
        return false;
    }
    if (kind == K_DISTANCE) return Sb(st + 3u) || P(pt + PT_P0) == 0u;
    if (kind == K_ALL_CORNER_RANDOM) return Sb(st + 2u);
    if (kind == K_FINAL_CORNER_LATER_HALF) return Sb(st + 1u);
    return true;
}

// check_activate: the terms with own state, then check_pure. s is the owner's skill.
bool check_activate(uint owner, uint s, uint pt, uint st, bool was_activated, bool others_activated) {
    uint kind = P(pt + PT_KIND), op = P(pt + PT_OP);
    int rh = Pi(pt + PT_RH);
    if (kind == K_RANDOM_LOT_OTHER_ACTIVATE) {
        if (Sb(st + 1u) && others_activated) return true;
        Ssetb(st + 1u, false);
        if (Sf(st + 2u) <= 0.0 && others_activated) {
            float rnd = race_random_float(0u, 100.0);
            float wiz = stat_wiz_for(owner, WIZ_SKILLS);
            Ssetf(st + 2u, Pf(pt + PT_P0));
            float per = float(double(p_lot_activate_per_val1()) - div_rn64(double(p_lot_activate_per_val2()), double(wiz)));
            if (!(float(p_activate_per_min()) <= per)) per = float(p_activate_per_min());
            bool ok = rnd < per;
            Ssetb(st + 1u, ok);
            return ok;
        }
        return false;
    }
    if (kind == K_HP_EMPTY_ONE_TIME) {
        if (!Sb(st + 2u)) {
            if (!h_is_param_initialized(owner)) return false;
            Ssetf(st + 1u, h_hp(owner));
            Ssetb(st + 2u, true);
        }
        if (Sf(st + 1u) <= 0.0) return false;
        Ssetf(st + 1u, h_hp(owner));
        return Sf(st + 1u) <= 0.0;
    }
    // change_order stays true once true; its value check is check_pure's.
    if (kind == K_CHANGE_ORDER && was_activated) return true;
    if (kind == K_SAME_SKILL_HORSE_COUNT) {
        if (!Sb(st + 2u)) {
            int id = skill_id(owner, s);
            for (uint j = 0u; j < uint(r_num_horses()); j++) {
                if (!h_is_skill_initialized(j)) return false;
                if (has_skill(j, id)) Sseti(st + 1u, Si(st + 1u) + 1);
            }
            Ssetb(st + 2u, true);
        }
        return check_value(Si(st + 1u), op, rh);
    }
    if (kind == K_ACTIVATE_OTHER_SKILL_DETAIL) {
        if (Sb(st + 1u)) return true;
        int id = skill_id(owner, s);
        uint prev = P(ph(owner) + PH_PREV_DETAILS);
        for (uint k = 0u; k < list_count(prev); k++) {
            if (skill_id(owner, list_get(prev, k) / 2u) == id) {
                Ssetb(st + 1u, true);
                break;
            }
        }
        return Sb(st + 1u);
    }
    if (kind == K_IS_ACTIVATE_ANY_SKILL) {
        int cur = activate_skill_count(owner) - Si(st + 2u);
        int prev = Si(st + 1u);
        Sseti(st + 1u, cur);
        if (prev < cur) Sseti(st + 2u, Si(st + 2u) + 1);
        return check_value(prev < cur ? 1 : 0, op, rh);
    }
    if (kind == K_LAST_STRAIGHT_ONE_TIME) {
        if (!Sb(st + 1u)) {
            bool v = h_is_straight(owner) && h_is_final_corner(owner);
            Ssetb(st + 1u, v);
            return v;
        }
        return false;
    }
    return check_pure(owner, pt, st);
}

// trigger_update: one term. Returns its is_activated; stores it where something reads it later.
bool trigger_update(uint owner, uint s, uint pt, bool keeps, float dt, bool reserved, bool others_activated) {
    uint kind = P(pt + PT_KIND), st = P(pt + PT_STATE);
    if (kind == K_NULL) return Sb(st);
    bool act = check_activate(owner, s, pt, st, Sb(st), others_activated);
    if (reserved) act = true;
    if (keeps) Ssetb(st, act);
    if (kind == K_DISTANCE && P(pt + PT_P0) != 0u && !Sb(st + 3u)) {
        // The retried windows (corner, final corner) fail again: the course is fixed.
    } else if (kind == K_ALL_CORNER_RANDOM && !Sb(st + 2u)) {
        gpu_fault(FAULT_SKILLS);  // initialized at construction
    } else if (kind == K_RANDOM_LOT_OTHER_ACTIVATE) {
        if (Sf(st + 2u) > 0.0) Ssetf(st + 2u, Sf(st + 2u) - dt);
    } else if (kind == K_FINAL_CORNER_LATER_HALF && !Sb(st + 1u)) {
        Ssetb(st + 1u, true);
        if (P(pt + PT_P0) != 0u) {
            Ssetb(st + 2u, true);
            Ssetf(st + 3u, Pf(pt + PT_P1));
            Ssetf(st + 4u, Pf(pt + PT_P2));
        } else {
            Ssetb(st + 2u, false);
        }
    }
    return act;
}

// collection_update: OR of AND-groups, left to right. Pure terms after a failed term of the group are skipped (as
// the CPU does), unless a term is random_lot_other_activate (which reads the others' last values).
bool collection_update(uint owner, uint s, uint first, uint count, uint flags, float dt, bool reserved) {
    if ((flags & 1u) == 0u) return false;
    bool any_empty = (flags & 2u) != 0u, lot_other = (flags & 4u) != 0u;
    bool any = any_empty;
    uint start = 0u;
    bool group_failed = false;
    for (uint k = 0u; k < count; k++) {
        uint pt = first + k * PT_WORDS;
        uint g = P(pt + PT_GROUP);
        if (P(first + start * PT_WORDS + PT_GROUP) != g) {
            if (!group_failed) any = true;
            start = k;
            group_failed = false;
        }
        uint st = P(pt + PT_STATE);
        if (group_failed && !lot_other && term_is_pure(pt, st)) continue;
        bool others = false;
        if (P(pt + PT_KIND) == K_RANDOM_LOT_OTHER_ACTIVATE) {
            others = true;
            for (uint j = start; j < count && P(first + j * PT_WORDS + PT_GROUP) == g; j++) {
                if (j != k && !Sb(P(first + j * PT_WORDS + PT_STATE))) others = false;
            }
        }
        bool keeps = lot_other || P(pt + PT_KIND) == K_CHANGE_ORDER;
        bool act = trigger_update(owner, s, pt, keeps, dt, reserved, others);
        if (!act) group_failed = true;
    }
    if (count > 0u && !group_failed) any = true;
    return any;
}

// ---- abilities

// The modifier ids an ability pushed: runs (first, last).
void ability_note_id(uint aa, uint id) {
    uint n = S(aa + AA_RUN_COUNT);
    if (n > 0u && S(aa + AA_RUNS + 2u * (n - 1u) + 1u) + 1u == id) {
        Sset(aa + AA_RUNS + 2u * (n - 1u) + 1u, id);
        return;
    }
    if (n >= MAX_RUNS) {
        gpu_fault(FAULT_SKILLS);
        return;
    }
    Sset(aa + AA_RUNS + 2u * n, id);
    Sset(aa + AA_RUNS + 2u * n + 1u, id);
    Sset(aa + AA_RUN_COUNT, n + 1u);
}

void push_modifier(uint aa, uint h, int param, bool set_kind, bool one_time, float v, float time) {
    uint lo = r_next_modifier_id(0u), hi = r_next_modifier_id(1u);
    if (hi != 0u) gpu_fault(FAULT_SKILLS);
    set_r_next_modifier_id(0u, lo + 1u);
    if (lo == 0xffffffffu) set_r_next_modifier_id(1u, hi + 1u);
    uint flags = (set_kind ? MOD_SET : 0u) | (one_time ? MOD_ONE_TIME : 0u) | (!one_time && 0.0 > time ? MOD_INFINITE : 0u);
    add_modifier(h, Modifier(param, flags, one_time ? 0.0 : time, v, lo, hi));
    ability_note_id(aa, lo);
}

// Targets of an ability (target.rs), into sel[0..n).
uint sel[MAX_HORSES];
uint sel_n;

void sel_push(uint h) {
    if (sel_n < MAX_HORSES) sel[sel_n++] = h;
    else gpu_fault(FAULT_SKILLS);
}
bool sel_contains(uint h) {
    for (uint k = 0u; k < sel_n; k++) {
        if (sel[k] == h) return true;
    }
    return false;
}

void select_targets(uint owner, int type, int v) {
    sel_n = 0u;
    uint n = uint(r_num_horses());
    if (type == 1) {
        sel_push(owner);
    } else if (type == 2 || type == 3) {
        for (uint h = 0u; h < n; h++) {
            if (type == 2 || h != owner) sel_push(h);
        }
    } else if (type == 4) {
        // The nearest visible horses in world space, stable by distance.
        int vc = h_visible_horse_count(owner);
        int m = vc < v ? vc : v;
        float dist[MAX_HORSES];
        for (int k = 0; k < m; k++) {
            uint h = uint(h_visible_horses(owner, uint(k)));
            float dx = h_position(owner, 0u) - h_position(h, 0u);
            float dy = h_position(owner, 1u) - h_position(h, 1u);
            float dz = h_position(owner, 2u) - h_position(h, 2u);
            float s = (dy * dy + dx * dx) + dz * dz;
            dist[sel_n] = float(sqrt_rn64(double(s)));
            sel_push(h);
        }
        for (uint i = 1u; i < sel_n; i++) {
            for (uint j = i; j > 0u && float_compare_to(dist[j - 1u], dist[j]) > 0; j--) {
                float t = dist[j];
                dist[j] = dist[j - 1u];
                dist[j - 1u] = t;
                uint u = sel[j];
                sel[j] = sel[j - 1u];
                sel[j - 1u] = u;
            }
        }
    } else if (type == 5) {
        // Random others: each pick starts at a random index and takes the next horse not yet picked.
        int hn = Pi(PR_FIELD_HORSE_NUM);
        int cnt = v >= hn - 1 ? hn - 1 : v;
        for (int c = 0; c < cnt; c++) {
            int r = race_random_int(0u, hn);
            for (int j = 0; j < hn; j++) {
                uint idx = uint((j + r) % hn);
                if (idx != owner && !sel_contains(idx)) {
                    sel_push(idx);
                    break;
                }
            }
        }
    } else if (type == 6) {
        for (uint h = 0u; h < n; h++) {
            if (h_cur_order(h) + 1 == v) {
                sel_push(h);
                break;
            }
        }
    } else if (type == 7 || type == 8) {
        for (uint h = 0u; h < n; h++) {
            if (type == 7 ? h_cur_order(h) + 1 < v : v < h_cur_order(h) + 1) sel_push(h);
        }
    } else if (type == 9) {
        int o = h_cur_order(owner);
        if (o != 0) {
            for (int k = 0; k < v; k++) {
                o--;
                if (o < 0) break;
                int h = horse_at_order(o);
                if (h >= 0) sel_push(uint(h));
            }
        }
    } else if (type == 10) {
        int o = h_cur_order(owner);
        if (r_last_order() != o) {
            for (int k = 0; k < v; k++) {
                o++;
                if (r_last_order() < o) break;
                int h = horse_at_order(o);
                if (h >= 0) sel_push(uint(h));
            }
        }
    } else if (type == 11) {
        if (team_id(owner) == 0) {
            sel_push(owner);
        } else {
            uint t = P(ph(owner) + PH_TEAM);
            for (uint k = 0u; k < P(ph(owner) + PH_TEAM_COUNT); k++) sel_push(P(t + k));
        }
    } else if (type == 12) {
        // Near horses by lane gap * 18 + distance gap (the unstable sort), the first v.
        int cnt = h_near_horse_count(owner);
        int m = v < cnt ? v : cnt;
        for (int k = 0; k < cnt; k++) {
            uint h = uint(h_near_horses(owner, uint(k)));
            sort_h[k] = int(h);
            sort_key[h] = h_ar_lane_gap_abs(owner, h) * 18.0 + h_ar_distance_gap_abs(owner, h);
        }
        sort_mode = SORT_KEY_FLOAT;
        if (!intro_sort(cnt)) gpu_fault(FAULT_SKILLS);
        for (int k = 0; k < m; k++) sel_push(uint(sort_h[k]));
    } else if (type == 13) {
        sel_push(owner);
        if (h_block_horse(owner) >= 0) sel_push(uint(h_block_horse(owner)));
    } else if (type == 14) {
        if (h_block_horse_in(owner) >= 0) sel_push(uint(h_block_horse_in(owner)));
        if (h_block_horse_out(owner) >= 0) sel_push(uint(h_block_horse_out(owner)));
    } else if (type == 15 || type == 16) {
        for (int k = 0; k < h_near_horse_count(owner); k++) {
            uint h = uint(h_near_horses(owner, uint(k)));
            float g = h_ar_distance_gap(owner, h);
            if (type == 15 ? g > 0.0 : 0.0 > g) sel_push(h);
        }
    } else if (type == 17 || type == 18 || type == 21) {
        uint style = uint(v & 0xff);
        if (style >= 1u && style <= 4u) {
            for (int k = 0; k < r_style_len(style); k++) {
                uint h = uint(r_style_horse(style * MAX_HORSES + uint(k)));
                if (type == 18 && h == owner) continue;
                if (type == 21 && !h_is_temptation(h)) continue;
                sel_push(h);
            }
        }
    } else if (type == 19 || type == 20) {
        int o = h_cur_order(owner);
        bool infront = type == 19;
        if (!((infront && o == 0) || (!infront && r_last_order() == o))) {
            uint cand[MAX_HORSES];
            uint cn = 0u;
            for (int k = 0; k < r_rushed_len(); k++) {
                uint h = uint(r_rushed_horse(uint(k)));
                if (infront ? h_cur_order(h) < o : o < h_cur_order(h)) cand[cn++] = h;
            }
            int m = int(cn) < v ? int(cn) : v;
            for (int c = 0; c < m; c++) {
                uint r = uint(race_random_int(0u, int(cn)));
                sel_push(cand[r]);
                for (uint j = r; j + 1u < cn; j++) cand[j] = cand[j + 1u];
                cn--;
            }
        }
    } else if (type == 22) {
        for (uint h = 0u; h < n; h++) {
            if (Pi(ph(h) + PH_CHARA_ID) == v) sel_push(h);
        }
    } else if (type == 23) {
        uint heal = P(PR_HEAL_LIST);
        for (int k = int(list_count(heal)) - 1; k >= 0; k--) {
            uint h = list_get(heal, uint(k));
            if (!sel_contains(h)) {
                sel_push(h);
                if (int(sel_n) == v) break;
            }
        }
    }
}

// calc_target: types 1..=23; a debuff drops horses with debuff cancel (the owner too) and the owner's teammates.
void calc_targets(uint owner, uint pa) {
    int type = Pi(pa + PA_TARGET_TYPE);
    sel_n = 0u;
    if (type < 1 || type > 23) return;
    select_targets(owner, type, Pi(pa + PA_TARGET_VALUE));
    if (P(pa + PA_TARGET_IS_DEBUFF) != 0u) {
        uint w = 0u;
        int a = team_id(owner);
        for (uint k = 0u; k < sel_n; k++) {
            uint h = sel[k];
            if (apply_modifier_bool(h, MOD_DEBUFF_CANCEL, false)) continue;
            if (h != owner) {
                int b = team_id(h);
                if (a != 0 && b != 0 && a == b) continue;
            }
            sel[w++] = h;
        }
        sel_n = w;
    }
}

// value.rs ability_value: read once per activation (usages 8 / 9 draw).
float ability_value(uint owner, uint pa) {
    if (P(pa + PA_IS_FIXED) != 0u) return Pf(pa + PA_FIXED_VALUE);
    uint usage = P(pa + PA_USAGE);
    float base = Pf(pa + PA_BASE);
    uint table = P(pa + PA_TABLE), tn = P(pa + PA_TABLE_COUNT);
    float mul = 1.0;
    if (usage == 8u || usage == 9u) {
        uint arr = P(usage == 8u ? PR_RANDOM1 : PR_RANDOM2), an = P(usage == 8u ? PR_RANDOM1_COUNT : PR_RANDOM2_COUNT);
        int total = 0;
        for (uint k = 0u; k < an; k++) total += int(Pf(arr + 2u * k + 1u));
        int v = race_random_int(0u, total);
        for (uint k = 0u; k < an; k++) {
            v -= int(Pf(arr + 2u * k + 1u));
            if (v < 1) {
                mul = Pf(arr + 2u * k);
                break;
            }
        }
    } else if (usage == 11u) {
        mul = asc_lookup(table, tn, float(h_ocu_corner_phase_end_after(owner)));
    } else if (usage == 14u) {
        mul = asc_lookup(table, tn, float(h_act_tag_group_count(owner)));
    } else if (usage == 15u) {
        mul = mathf_min(float(h_act_heal_count(owner)) * uintBitsToFloat(0x3dcccccdu) + 1.0, 1.5);  // 0.1f
    } else if (usage == 16u) {
        int o = h_final_corner_end_order(owner);
        mul = o == -1 ? 1.0 : mathf_max(uintBitsToFloat(0x3f99999au) - float(o) * uintBitsToFloat(0x3ca3d70au), 1.0);  // 1.2f, 0.02f
    } else if (usage == 19u) {
        float diff = h_distance(first_horse()) - h_distance(owner);
        float add = Pf(pa + PA_FIXED_VALUE);
        for (uint k = 0u; k < tn; k++) {
            if (diff < Pf(table + 2u * k)) {
                add = Pf(table + 2u * k + 1u);
                break;
            }
        }
        return add + base;
    } else if (usage == 20u || usage == 21u) {
        mul = asc_lookup(table, tn, h_block_side_max_by_phase(owner, 1u));
    } else if (usage == 22u || usage == 23u) {
        mul = asc_lookup(table, tn, stat_speed(owner));
    } else {
        gpu_fault(FAULT_SKILLS);
    }
    return mul * base;
}

// ability::activate: targets (may draw), value (may draw), the additional-activation gate, then the effects.
void ability_activate(uint owner, uint det, uint pa, bool is_additional) {
    calc_targets(owner, pa);
    uint aa = P(pa + PA_STATE);
    float v = ability_value(owner, pa);
    Ssetf(aa + AA_VALUE_ON_ACTIVATE, v);
    if (P(pa + PA_AA_TYPE) == 0u) {
        if (is_additional) Sset(aa + AA_ADDITIONAL_ACTIVATE_COUNT, S(aa + AA_ADDITIONAL_ACTIVATE_COUNT) + 1u);
    } else {
        if (!is_additional) return;
        Sset(aa + AA_ADDITIONAL_ACTIVATE_COUNT, S(aa + AA_ADDITIONAL_ACTIVATE_COUNT) + 1u);
    }
    int type = Pi(pa + PA_TYPE);
    float time = Sf(P(det + PD_STATE) + AD_LEFT_TIME);
    // The modifiers of the type, pushed below by one push_modifier call (inlined once): n of them, the first with
    // param, set_kind, one_time, value and time; 28 adds the lane-move flag, 32 is every stat in order.
    int param = -1, n = 1;
    bool set_kind = false, one_time = false;
    float mv = v, mt = time;
    if (type >= 1 && type <= 5) param = MOD_SPEED + (type - 1);
    else if (type == 6) { param = MOD_RUNNING_STYLE_EX_OONIGE; set_kind = true; mv = 1.0; }
    else if (type == 7) { param = MOD_HP_DEC_RATE; set_kind = true; }
    else if (type == 8) param = MOD_VISIBLE_DISTANCE;
    else if (type == 9) { param = MOD_HP; one_time = true; mt = 0.0; }
    else if (type == 10) { param = MOD_START_DELAY_SCALE; set_kind = true; one_time = true; mv = v < 0.0 ? 0.0 : v; mt = 0.0; }
    else if (type == 11) { param = MOD_FORCE_OVERTAKE_IN; set_kind = true; mv = 1.0; }
    else if (type == 12) { param = MOD_FORCE_OVERTAKE_OUT; set_kind = true; mv = 1.0; }
    else if (type == 13) { param = MOD_TEMPTATION_END_TIME; set_kind = true; one_time = true; mt = 0.0; }
    else if (type == 14) { param = MOD_START_DELAY_FIX; set_kind = true; one_time = true; mt = 0.0; }
    else if (type == 21) param = MOD_CURRENT_SPEED;
    else if (type == 22) param = MOD_CURRENT_SPEED_WITH_NATURAL_DECELERATION;
    else if (type == 27) param = MOD_TARGET_SPEED;
    else if (type == 28) { param = MOD_LANE_MOVE_SPEED; n = 0.0 < v ? 2 : 1; }
    else if (type == 29) param = MOD_TEMPTATION_PER;
    else if (type == 30) param = MOD_PUSH_PER;
    else if (type == 31) param = MOD_ACCEL;
    else if (type == 32) { param = MOD_SPEED; n = MOD_WIZ - MOD_SPEED + 1; }
    else if (type == 35) param = MOD_TARGET_LANE;
    else if (type == 36) { param = MOD_ACTIVATE_RANDOM_NORMAL_AND_RARE_SKILL; set_kind = true; one_time = true; mt = 0.0; }
    else if (type == 37) { param = MOD_ACTIVATE_RANDOM_RARE_SKILL; set_kind = true; one_time = true; mt = 0.0; }
    else if (type == 38) { param = MOD_DEBUFF_CANCEL; set_kind = true; mv = 1.0; }
    if (param < 0) return;
    for (uint k = 0u; k < sel_n; k++) {
        uint h = sel[k];
        if (type == 14 && has_modifier(h, MOD_START_DELAY_FIX)) {
            // The higher StartDelayFix wins.
            if (apply_modifier(h, MOD_START_DELAY_FIX, 0.0) > v) continue;
            remove_modifier(h, MOD_START_DELAY_FIX);
        }
        for (int j = 0; j < n; j++) {
            int pj = param + (type == 32 ? j : 0);
            bool sk = set_kind;
            float vj = type == 9 ? h_max_hp(h) * v : mv;
            if (type == 28 && j == 1) {
                pj = MOD_TARGET_SPEED_ON_MOVE_LANE;
                sk = true;
                vj = 1.0;
            }
            push_modifier(aa, h, pj, sk, one_time, vj, mt);
        }
    }
}

// add_ability_time on every modifier the ability pushed that is still in a pool.
void ability_add_time(uint pa, float extra) {
    uint aa = P(pa + PA_STATE);
    uint runs = S(aa + AA_RUN_COUNT);
    if (runs == 0u) return;
    for (uint h = 0u; h < uint(r_num_horses()); h++) {
        for (uint k = 0u; k < uint(h_mod_count(h)); k++) {
            if (h_mod_id_hi(h, k) != 0u) continue;
            uint id = h_mod_id_lo(h, k);
            for (uint r = 0u; r < runs; r++) {
                if (S(aa + AA_RUNS + 2u * r) <= id && id <= S(aa + AA_RUNS + 2u * r + 1u)) {
                    uint f = h_mod_flags(h, k);
                    if ((f & (MOD_INFINITE | MOD_ONE_TIME)) == 0u) set_h_mod_time(h, k, extra + h_mod_time(h, k));
                }
            }
        }
    }
}

// Additional activations earned this frame (counted against the cap).
int calc_additional_activate_count(uint owner, uint s, uint pa) {
    uint aa = P(pa + PA_STATE);
    uint kind = P(pa + PA_AA_KIND);
    int mx = Pi(pa + PA_AA_MAX), count = Si(aa + AA_AA_COUNT);
    if (kind == 1u) {
        if (h_ocu_changed(owner) && count < mx) {
            Sseti(aa + AA_AA_COUNT, count + 1);
            return 1;
        }
        return 0;
    }
    if (kind == 2u) {
        uint prev = P(ph(owner) + PH_PREV_DETAILS);
        int prev_count = int(list_count(prev));
        int remain = mx - count;
        if (remain < 1 || prev_count < 1) return 0;
        int id = skill_id(owner, s);
        bool has_self = false;
        for (uint k = 0u; k < uint(prev_count); k++) {
            if (skill_id(owner, list_get(prev, k) / 2u) == id) has_self = true;
        }
        int n = has_self ? prev_count - 1 : prev_count;
        if (n < remain) remain = n;
        Sseti(aa + AA_AA_COUNT, count + remain);
        return remain;
    }
    return 0;
}

// ---- detail lifecycle (mod.rs)

// value.rs ability_time: the duration, read when the detail activates.
float ability_time(uint owner, uint det) {
    uint usage = P(det + PD_TIME_USAGE);
    float base = Pf(det + PD_TIME_BASE);
    uint table = P(det + PD_TIME_TABLE), tn = P(det + PD_TIME_TABLE_COUNT);
    float mul;
    if (usage == 2u) {
        float per = Pf(table), mn = Pf(table + 1u), mx = Pf(table + 2u);
        float v = div_rn32(h_distance(first_horse()) - h_distance(owner), per) + mn;
        mul = mathf_clamp(v, mn, mx);
    } else if (usage == 3u || usage == 7u) {
        mul = asc_lookup(table, tn, h_hp(owner));
    } else if (usage == 5u || usage == 6u) {
        mul = asc_lookup(table, tn, h_block_side_max_by_phase(owner, 1u));
    } else {
        return base;
    }
    return mul * base;
}

bool detail_is_reserved(uint owner, uint s, uint d) {
    return d == 0u && list_contains(P(ph(owner) + PH_RESERVED_IDS), uint(skill_id(owner, s)));
}

void add_prev_activate_detail(uint owner, uint s, uint d) {
    uint prev = P(ph(owner) + PH_PREV_DETAILS);
    if (!list_contains(prev, detail_code(s, d))) list_push(prev, detail_code(s, d));
}

void detail_activate(uint owner, uint s, uint d) {
    uint det = pd(owner, s, d), a = P(det + PD_STATE);
    Ssetb(a + AD_IS_ACTIVATED, true);
    Ssetf(a + AD_LEFT_TIME, ability_time(owner, det));
    uint n = P(det + PD_ABILITY_COUNT);
    for (uint k = 0u; k < n; k++) {
        uint pa = P(det + PD_ABILITIES) + k * PA_WORDS;
        if (P(pa + PA_PRESENT) != 0u) ability_activate(owner, det, pa, false);
    }
    if (P(det + PD_IS_NOT_ACTIVATE_SKILL) != 0u) return;
    uint acts = P(ph(owner) + PH_ACTIVATE_DETAILS);
    if (!list_contains(acts, detail_code(s, d))) list_push(acts, detail_code(s, d));
    uint used = P(ph(owner) + PH_USED_IDS);
    uint id = uint(skill_id(owner, s));
    if (!list_contains(used, id)) list_push(used, id);
    int phase = h_phase(owner);
    if (phase >= 0 && phase < 4) set_h_act_count_by_phase(owner, uint(phase), h_act_count_by_phase(owner, uint(phase)) + 1);
    uint later_half = float(r_course_distance()) * 0.5 <= h_distance(owner) ? 1u : 0u;
    set_h_act_count_by_distance(owner, later_half, h_act_count_by_distance(owner, later_half) + 1);
    bool heal = false;
    for (uint k = 0u; k < n; k++) {
        uint pa = P(det + PD_ABILITIES) + k * PA_WORDS;
        if (P(pa + PA_PRESENT) != 0u && Pi(pa + PA_TYPE) == 9 && Sf(P(pa + PA_STATE) + AA_VALUE_ON_ACTIVATE) > 0.0) heal = true;
    }
    if (heal) set_h_act_heal_count(owner, h_act_heal_count(owner) + 1);
    if (P(ps(owner, s) + PS_IS_SPECIFIC) != 0u) set_h_act_tag_group_count(owner, h_act_tag_group_count(owner) + 1);
}

// Runs an activated detail; true when it ended this frame.
bool detail_update_skill(uint owner, uint s, uint d, float dt) {
    uint det = pd(owner, s, d), a = P(det + PD_STATE);
    bool infinite = P(det + PD_IS_TIME_INFINITE) != 0u;
    if (Sb(a + AD_IS_ACTIVATED)) {
        for (uint k = 0u; k < P(det + PD_ABILITY_COUNT); k++) {
            uint pa = P(det + PD_ABILITIES) + k * PA_WORDS;
            if (P(pa + PA_PRESENT) == 0u) continue;
            int n = calc_additional_activate_count(owner, s, pa);
            for (int j = 0; j < n; j++) ability_activate(owner, det, pa, true);
        }
        if (!infinite) {
            float add = 0.0;
            if (P(det + PD_TIME_USAGE) == 4u) {
                // update_ability_time: each frame the horse gains a position adds time, up to a count.
                uint table = P(det + PD_TIME_TABLE);
                if (!h_ocu_changed(owner) || Pf(table) <= float(Si(a + AD_INCREMENT_COUNT))) {
                    Ssetf(a + AD_ADDITIONAL_TIME, 0.0);
                } else {
                    Sseti(a + AD_INCREMENT_COUNT, Si(a + AD_INCREMENT_COUNT) + 1);
                    Ssetf(a + AD_ADDITIONAL_TIME, Pf(table + 1u));
                }
                add = Sf(a + AD_ADDITIONAL_TIME);
            }
            if (0.0 < add) {
                Ssetf(a + AD_LEFT_TIME, add + Sf(a + AD_LEFT_TIME));
                for (uint k = 0u; k < P(det + PD_ABILITY_COUNT); k++) {
                    uint pa = P(det + PD_ABILITIES) + k * PA_WORDS;
                    if (P(pa + PA_PRESENT) != 0u) ability_add_time(pa, add);
                }
            }
        }
    }
    if (infinite) return false;
    float left = Sf(a + AD_LEFT_TIME) - dt;
    Ssetf(a + AD_LEFT_TIME, left);
    if (0.0 < left) return false;
    detail_stop(owner, s, d);
    return true;
}

// Checks the triggers and activates the detail. The precondition latches. One collection_update call for both
// (precondition, then condition), so it is inlined once.
bool detail_update_trigger_and_activate(uint owner, uint s, uint d, float dt) {
    bool reserved = detail_is_reserved(owner, s, d);
    uint det = pd(owner, s, d), a = P(det + PD_STATE);
    bool check_pre = !reserved && Sb(a + AD_NEED_PRE);
    if (!check_pre && (P(det + PD_MAIN_FLAGS) & 1u) == 0u) return false;
    for (uint slot = check_pre ? 0u : 1u; slot < 2u; slot++) {
        bool pre = slot == 0u;
        uint first = P(det + (pre ? PD_PRE : PD_MAIN));
        uint count = P(det + (pre ? PD_PRE_COUNT : PD_MAIN_COUNT));
        uint flags = P(det + (pre ? PD_PRE_FLAGS : PD_MAIN_FLAGS));
        if (!collection_update(owner, s, first, count, flags, dt, reserved)) return false;
        if (pre) Ssetb(a + AD_NEED_PRE, false);
    }
    detail_activate(owner, s, d);
    return true;
}

bool detail_check_cool_down(uint a, float dt) {
    float cd = Sf(a + AD_COOL_DOWN_TIME);
    if (0.0 < cd) {
        cd -= dt;
        if (cd < 0.0) cd = 0.0;
        Ssetf(a + AD_COOL_DOWN_TIME, cd);
    }
    return cd <= 0.0;
}

// update_awake_skill: active details run; otherwise the skill cooldown ticks and the first detail whose trigger fires
// activates. Details with their own cooldown are handled at the end. Written as one loop over (pass, detail) with one
// call each of detail_update_skill and detail_update_trigger_and_activate, so each is inlined once.
void skill_update(uint owner, uint s, float dt) {
    uint prev = P(ph(owner) + PH_PREV_DETAILS);
    if (h_is_complete_first_update(owner) && list_count(prev) > 0u) {
        for (uint d = 0u; d < 2u; d++) {
            for (uint k = 0u; k < list_count(prev); k++) {
                if (list_get(prev, k) == detail_code(s, d)) {
                    list_remove_at(prev, k);
                    break;
                }
            }
        }
    }
    bool any_normal_active = false;
    for (uint d = 0u; d < 2u; d++) {
        uint det = pd(owner, s, d);
        if (P(det + PD_IS_USE_DETAIL_COOL_DOWN) == 0u && Sb(P(det + PD_STATE) + AD_IS_ACTIVATED)) any_normal_active = true;
    }
    uint sst = skill_state(owner, s);
    bool enable = Sb(sst + 1u);
    bool ready = false;
    if (!any_normal_active) {
        float cd = Sf(sst);
        if (0.0 < cd) {
            cd -= dt;
            if (cd < 0.0) cd = 0.0;
            Ssetf(sst, cd);
        }
        ready = cd <= 0.0;
    }
    // Pass 0: the details without their own cooldown (run the active ones, or try each until one activates).
    // Pass 1: the details with their own cooldown.
    bool activated_one = false;
    for (uint it = 0u; it < 4u; it++) {
        uint d = it & 1u;
        bool own_cd = it >= 2u;
        uint det = pd(owner, s, d), a = P(det + PD_STATE);
        if ((P(det + PD_IS_USE_DETAIL_COOL_DOWN) != 0u) != own_cd) continue;
        bool is_active = Sb(a + AD_IS_ACTIVATED);
        bool run = false, trigger = false;
        if (!own_cd) {
            if (any_normal_active) {
                run = is_active;
            } else if (ready && !activated_one && (enable || detail_is_reserved(owner, s, d))) {
                trigger = !is_active;
            }
        } else if (is_active) {
            run = true;
        } else {
            trigger = detail_check_cool_down(a, dt);
        }
        if (run) {
            if (detail_update_skill(owner, s, d, dt) && !own_cd) Ssetf(sst, Pf(det + PD_DEFAULT_COOL_DOWN_TIME));
        } else if (trigger && detail_update_trigger_and_activate(owner, s, d, dt)) {
            if (!own_cd) activated_one = true;
            if (!own_cd || P(det + PD_IS_NOT_ACTIVATE_SKILL) == 0u) add_prev_activate_detail(owner, s, d);
        }
    }
}

// Re-draws the skills reserved for random activation (ability types 36 / 37): per modifier, up to n eligible skills.
void update_random_activate_skill_id_list(uint owner) {
    uint res = P(ph(owner) + PH_RESERVED_IDS), acts = P(ph(owner) + PH_ACTIVATE_DETAILS);
    list_clear(res);
    for (uint pass = 0u; pass < 2u; pass++) {
        int param = pass == 0u ? MOD_ACTIVATE_RANDOM_RARE_SKILL : MOD_ACTIVATE_RANDOM_NORMAL_AND_RARE_SKILL;
        int n = f32_to_i32(apply_modifier(owner, param, 0.0));
        if (n <= 0) continue;
        int cand[64];
        int cn = 0;
        for (uint s = 0u; s < skill_count(owner); s++) {
            uint sp = ps(owner, s);
            int rarity = Pi(sp + PS_RARITY), id = Pi(sp + PS_ID);
            bool rare_ok = rarity == 2 || rarity == 6 || (pass == 1u && rarity == 1);
            if (!rare_ok || P(sp + PS_TIME_MINUS_ONE) != 0u || !(Sf(skill_state(owner, s)) <= 0.0) || Pi(sp + PS_IS_GENERAL) != 1) continue;
            bool in_use = false;
            for (uint k = 0u; k < list_count(acts); k++) {
                if (skill_id(owner, list_get(acts, k) / 2u) == id) in_use = true;
            }
            if (in_use || list_contains(res, uint(id))) continue;
            if (cn >= 64) {
                gpu_fault(FAULT_SKILLS);
                break;
            }
            cand[cn++] = id;
        }
        int drawn = n < cn ? n : cn;
        for (int c = 0; c < drawn; c++) {
            int k = net_random_next_range(0u, 0, cn);
            list_push(res, uint(cand[k]));
            for (int j = k; j + 1 < cn; j++) cand[j] = cand[j + 1];
            cn--;
        }
    }
}

// The two update groups: priority 1 (SkillsFirst), then priority 0 (SkillsSecond, then the random reservation).
void skills_update_group(uint owner, int priority, float dt) {
    for (uint s = 0u; s < skill_count(owner); s++) {
        if (Pi(ps(owner, s) + PS_PRIORITY) == priority) skill_update(owner, s, dt);
    }
}

void skills_first_stage(float dt) {
    skills_begin();
    for (uint i = 0u; i < uint(r_num_horses()); i++) {
        if (!h_is_finished(i)) skills_update_group(i, 1, dt);
    }
}

void skills_second_stage(float dt) {
    skills_begin();
    for (uint i = 0u; i < uint(r_num_horses()); i++) {
        if (!h_is_finished(i)) {
            skills_update_group(i, 0, dt);
            update_random_activate_skill_id_list(i);
        }
    }
}

#endif
