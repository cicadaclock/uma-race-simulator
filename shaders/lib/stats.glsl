// physics/stats.rs: stats after skill modifiers, clamped to [1, 2000], and horse::calc_section. Needs exact.glsl,
// race_state.glsl and modifiers.glsl.

#ifndef UMA_STATS_GLSL
#define UMA_STATS_GLSL

// api::wiz_mechanic: which mechanics read the alternative Wit (wiz_alt_mechanics)
const uint WIZ_SECTIONS = 1u, WIZ_POSITION_KEEP = 2u, WIZ_LAST_SPURT = 4u, WIZ_SKILLS = 8u, WIZ_RUSHED = 16u;
const uint WIZ_DOWNHILL = 32u, WIZ_COURSE_BONUS = 64u;

float clamp_stat(float v) { return mathf_clamp(v, 1.0, 2000.0); }
float stat_speed(uint h) { return clamp_stat(apply_modifier(h, MOD_SPEED, h_base_speed_adjusted(h))); }
float stat_pow(uint h) { return clamp_stat(apply_modifier(h, MOD_POWER, h_base_pow_adjusted(h))); }
float stat_guts(uint h) { return clamp_stat(apply_modifier(h, MOD_GUTS, h_base_guts_adjusted(h))); }
// stats::wiz_for: live Wit as mechanic m reads it.
float stat_wiz_for(uint h, uint m) {
    float base = (h_wiz_alt_mechanics(h) & m) != 0u ? h_base_wiz_adjusted_alt(h) : h_base_wiz_adjusted(h);
    return clamp_stat(apply_modifier(h, MOD_WIZ, base));
}

// The 1-based course section (of 24) the horse is in.
int calc_section(uint h) { return floor_to_int(div_rn32(h_distance(h), r_course_section_distance())) + 1; }

#endif
