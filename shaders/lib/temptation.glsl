// ai/temptation.rs: the rush itself and its end rolls (the pre-race lottery runs before the first frame). Needs
// exact.glsl, race_state.glsl, net_random.glsl, queries.glsl and stats.glsl.

#ifndef UMA_TEMPTATION_GLSL
#define UMA_TEMPTATION_GLSL

float tm_tick(float t, float dt) {
    if (t > 0.0) {
        t -= dt;
        if (0.0 > t) t = 0.0;
    }
    return t;
}

// Ends the rush (the count is kept).
void tm_end(uint i) {
    set_h_is_temptation(i, false);
    set_h_tm_mode(i, TM_NULL);
}

// What the rush does for this running style: one GetRandom(100) for Senko / Sashi / Oikomi, none for Nige. The
// thresholds are cumulative f32 sums.
int tm_calc_mode(int style) {
    if (style == STYLE_NIGE) return TM_BOOST;
    if (style == STYLE_SENKO) {
        float r = float(race_random_int(0u, 100));
        return p_tm_senko_to_nige() > r ? TM_POSITION_NIGE : TM_NULL;
    }
    if (style == STYLE_SASHI) {
        float r = float(race_random_int(0u, 100));
        float acc = p_tm_sashi_to_senko();
        if (acc > r) return TM_POSITION_SENKO;
        acc = acc + p_tm_sashi_to_nige();
        return acc > r ? TM_POSITION_NIGE : TM_NULL;
    }
    if (style == STYLE_OIKOMI) {
        float r = float(race_random_int(0u, 100));
        float acc = p_tm_oikomi_to_sashi();
        if (acc > r) return TM_POSITION_SASHI;
        acc = acc + p_tm_oikomi_to_senko();
        if (acc > r) return TM_POSITION_SENKO;
        acc = acc + p_tm_oikomi_to_nige();
        return acc > r ? TM_POSITION_NIGE : TM_NULL;
    }
    return TM_NULL;
}

// Starts the rush at its section, then ends it on the force-end timer, a skill's end timer, or a periodic end roll.
void temptation_update(uint i, float dt) {
    if (!h_is_temptation(i)) {
        if (!h_tm_start_enable(i)) return;
        if (calc_section(i) < h_tm_start_section(i)) return;
        set_h_is_temptation(i, true);
        set_h_tm_mode(i, tm_calc_mode(h_running_style(i)));
        set_h_tm_end_check_sec(i, p_tm_end_check_time());
        set_h_temptation_count(i, h_temptation_count(i) + 1);
        set_h_tm_force_end_sec(i, p_tm_force_end_time());
        set_h_tm_skill_end_enable(i, false);
        set_h_tm_skill_end_time(i, 0.0);
        set_h_tm_start_enable(i, false);
        return;
    }
    if (!h_tm_skill_end_enable(i)) {
        float f = tm_tick(h_tm_force_end_sec(i), dt);
        set_h_tm_force_end_sec(i, f);
        if (!(0.0 < f)) {
            tm_end(i);
            return;
        }
        float c = tm_tick(h_tm_end_check_sec(i), dt);
        set_h_tm_end_check_sec(i, c);
        if (0.0 < c) return;
        set_h_tm_end_check_sec(i, p_tm_end_check_time());
        int r = race_random_int(0u, 100);
        if (p_tm_end_per() > float(r)) tm_end(i);
    } else {
        float t = tm_tick(h_tm_skill_end_time(i), dt);
        set_h_tm_skill_end_time(i, t);
        if (!(0.0 < t)) tm_end(i);
    }
}

#endif
