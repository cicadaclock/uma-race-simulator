// skill/modifier.rs SkillModifierReceiver on the modifier pool of the race state: one list per horse, ordered by
// parameter (SkillModifierParam), each parameter's entries in their Rust list order. Every operation only reads or
// reorders the entries of one parameter, so this order gives the Rust results, and it is canonical (gpu::pack
// writes the same order). Needs exact.glsl and race_state.glsl.

#ifndef UMA_MODIFIERS_GLSL
#define UMA_MODIFIERS_GLSL

const uint MOD_SET = 1u, MOD_ONE_TIME = 2u, MOD_INFINITE = 4u;

// SkillModifierParam
const int MOD_SPEED = 0, MOD_STAMINA = 1, MOD_POWER = 2, MOD_GUTS = 3, MOD_WIZ = 4, MOD_CURRENT_SPEED = 5;
const int MOD_TARGET_SPEED = 6, MOD_LANE_MOVE_SPEED = 7, MOD_TEMPTATION_PER = 8, MOD_PUSH_PER = 9, MOD_ACCEL = 10;
const int MOD_TARGET_LANE = 11, MOD_HP = 12, MOD_HP_DEC_RATE = 13, MOD_TEMPTATION_END_TIME = 14;
const int MOD_START_DELAY_SCALE = 15, MOD_FORCE_OVERTAKE_IN = 16, MOD_FORCE_OVERTAKE_OUT = 17, MOD_DEBUFF_CANCEL = 18;
const int MOD_VISIBLE_DISTANCE = 19, MOD_RUNNING_STYLE_EX_OONIGE = 20, MOD_TARGET_SPEED_ON_MOVE_LANE = 21;
const int MOD_START_DELAY_FIX = 22, MOD_CURRENT_SPEED_WITH_NATURAL_DECELERATION = 23;
const int MOD_ACTIVATE_RANDOM_NORMAL_AND_RARE_SKILL = 24, MOD_ACTIVATE_RANDOM_RARE_SKILL = 25;
const uint GPU_FAULT_MODIFIER_OVERFLOW = 8u;

struct Modifier {
    int param;
    uint flags;
    float time;
    float value;
    uint id_lo;
    uint id_hi;
};

Modifier mod_get(uint h, uint k) {
    return Modifier(h_mod_param(h, k), h_mod_flags(h, k), h_mod_time(h, k), h_mod_value(h, k), h_mod_id_lo(h, k), h_mod_id_hi(h, k));
}
void mod_put(uint h, uint k, Modifier m) {
    set_h_mod_param(h, k, m.param);
    set_h_mod_flags(h, k, m.flags);
    set_h_mod_time(h, k, m.time);
    set_h_mod_value(h, k, m.value);
    set_h_mod_id_lo(h, k, m.id_lo);
    set_h_mod_id_hi(h, k, m.id_hi);
}
const Modifier MOD_EMPTY = Modifier(0, 0u, 0.0, 0.0, 0u, 0u);

// Removes entry k, keeping the order, and clears the freed slot.
void mod_remove_at(uint h, uint k) {
    uint n = uint(h_mod_count(h));
    for (uint j = k; j + 1u < n; j++) mod_put(h, j, mod_get(h, j + 1u));
    mod_put(h, n - 1u, MOD_EMPTY);
    set_h_mod_count(h, int(n) - 1);
}

// SkillModifierReceiver::add_modifier: after the parameter's last entry.
void add_modifier(uint h, Modifier m) {
    uint n = uint(h_mod_count(h));
    if (n >= MAX_MODIFIERS) {
        set_r_gpu_fault(r_gpu_fault() | GPU_FAULT_MODIFIER_OVERFLOW);
        return;
    }
    uint at = n;
    while (at > 0u && h_mod_param(h, at - 1u) > m.param) at--;
    for (uint j = n; j > at; j--) mod_put(h, j, mod_get(h, j - 1u));
    mod_put(h, at, m);
    set_h_mod_count(h, int(n) + 1);
}

bool has_modifier(uint h, int t) {
    for (uint k = 0u; k < uint(h_mod_count(h)); k++) {
        if (h_mod_param(h, k) == t) return true;
    }
    return false;
}

// Folds the parameter's modifiers over `value` in list order: Add adds, Set replaces.
float apply_modifier(uint h, int t, float value) {
    for (uint k = 0u; k < uint(h_mod_count(h)); k++) {
        if (h_mod_param(h, k) != t) continue;
        value = (h_mod_flags(h, k) & MOD_SET) != 0u ? h_mod_value(h, k) : value + h_mod_value(h, k);
    }
    return value;
}

bool apply_modifier_bool(uint h, int t, bool value) {
    for (uint k = 0u; k < uint(h_mod_count(h)); k++) {
        if (h_mod_param(h, k) != t) continue;
        if ((h_mod_flags(h, k) & MOD_SET) != 0u) value = !mathf_approximately(h_mod_value(h, k), 0.0);
    }
    return value;
}

void remove_modifier(uint h, int t) {
    uint k = 0u;
    while (k < uint(h_mod_count(h))) {
        if (h_mod_param(h, k) == t) mod_remove_at(h, k);
        else k++;
    }
}

// SkillParamModifier::add_ability_time on the parameter's first entry with this id.
void add_ability_time(uint h, int t, uint id_lo, uint id_hi, float extra) {
    for (uint k = 0u; k < uint(h_mod_count(h)); k++) {
        if (h_mod_param(h, k) != t || h_mod_id_lo(h, k) != id_lo || h_mod_id_hi(h, k) != id_hi) continue;
        uint f = h_mod_flags(h, k);
        if ((f & (MOD_INFINITE | MOD_ONE_TIME)) == 0u) set_h_mod_time(h, k, extra + h_mod_time(h, k));
        return;
    }
}

void otm_push(uint h, Modifier m) {
    uint n = uint(h_otm_count(h));
    if (n >= MAX_ONE_TIME_MODIFIERS) {
        set_r_gpu_fault(r_gpu_fault() | GPU_FAULT_MODIFIER_OVERFLOW);
        return;
    }
    set_h_otm_flags(h, n, m.flags);
    set_h_otm_time(h, n, m.time);
    set_h_otm_value(h, n, m.value);
    set_h_otm_id_lo(h, n, m.id_lo);
    set_h_otm_id_hi(h, n, m.id_hi);
    set_h_otm_count(h, int(n) + 1);
}

// SkillModifierReceiver::update_modifiers: each list from its end; an expired modifier is removed, and moved to the
// one-time list when it is CurrentSpeedWithNaturalDeceleration. Walking the whole pool from its end visits each
// parameter's entries in that order.
void update_modifiers(uint h, float dt) {
    for (int k = h_mod_count(h) - 1; k >= 0; k--) {
        Modifier m = mod_get(h, uint(k));
        bool expired;
        if ((m.flags & MOD_INFINITE) != 0u) expired = false;
        else if ((m.flags & MOD_ONE_TIME) != 0u) expired = true;
        else {
            m.time = m.time - dt;
            if (0.0 > m.time) m.time = 0.0;
            set_h_mod_time(h, uint(k), m.time);
            expired = 0.0 >= m.time;
        }
        if (expired) {
            mod_remove_at(h, uint(k));
            if (m.param == MOD_CURRENT_SPEED_WITH_NATURAL_DECELERATION) otm_push(h, m);
        }
    }
}

// SkillModifierReceiver::apply_one_time_skill_modifier (CurrentSpeedWithNaturalDeceleration): folds and clears.
float apply_one_time_skill_modifier(uint h, float value) {
    uint n = uint(h_otm_count(h));
    if (n == 0u) return value;
    for (uint k = 0u; k < n; k++) {
        value = (h_otm_flags(h, k) & MOD_SET) != 0u ? h_otm_value(h, k) : value + h_otm_value(h, k);
        set_h_otm_flags(h, k, 0u);
        set_h_otm_time(h, k, 0.0);
        set_h_otm_value(h, k, 0.0);
        set_h_otm_id_lo(h, k, 0u);
        set_h_otm_id_hi(h, k, 0u);
    }
    set_h_otm_count(h, 0);
    return value;
}

#endif
