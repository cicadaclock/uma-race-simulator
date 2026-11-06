// The skill program and skill state buffers (built by gpu::skills; record formats in skill_layout.glsl), their
// records and lists, and the skill state changes of the other stages: skill::clear_skills (UpdateHorses, on the
// finish) and the EndOfFrame skill caches. Needs exact.glsl, race_state.glsl, queries.glsl and skill_layout.glsl.

#ifndef UMA_SKILL_STATE_GLSL
#define UMA_SKILL_STATE_GLSL

#ifndef SKILL_PROGRAM_BINDING
#define SKILL_PROGRAM_BINDING 3
#endif
#ifndef SKILL_STATE_BINDING
#define SKILL_STATE_BINDING 4
#endif

layout(std430, binding = SKILL_PROGRAM_BINDING) readonly buffer SkillProgram { uint skp[]; };
layout(std430, binding = SKILL_STATE_BINDING) buffer SkillState { uint sks[]; };

const uint FAULT_SKILLS = 256u;

uint sk_base;         // r_skill_base(), set by skills_begin()
uint sk_state_words;  // this race's state size

// Every access is bounds-checked: a bad offset faults instead of reading another race's words.
uint P(uint w) {
    if (sk_base + w >= uint(skp.length())) {
        gpu_fault(FAULT_SKILLS);
        return 0u;
    }
    return skp[sk_base + w];
}
int Pi(uint w) { return int(P(w)); }
float Pf(uint w) { return uintBitsToFloat(P(w)); }
uint S(uint w) {
    if (w >= sk_state_words) {
        gpu_fault(FAULT_SKILLS);
        return 0u;
    }
    return sks[w * st_lanes + st_lane];
}
int Si(uint w) { return int(S(w)); }
float Sf(uint w) { return uintBitsToFloat(S(w)); }
bool Sb(uint w) { return S(w) != 0u; }
void Sset(uint w, uint v) {
    if (w >= sk_state_words) {
        gpu_fault(FAULT_SKILLS);
        return;
    }
    sks[w * st_lanes + st_lane] = v;
}
void Sseti(uint w, int v) { Sset(w, uint(v)); }
void Ssetf(uint w, float v) { Sset(w, floatBitsToUint(v)); }
void Ssetb(uint w, bool v) { Sset(w, v ? 1u : 0u); }

void skills_begin() {
    sk_base = r_skill_base();
    sk_state_words = 0xffffffffu;
    sk_state_words = P(PR_STATE_WORDS);
    if ((P(PR_FLAGS) & 1u) != 0u) gpu_fault(FAULT_SKILLS);
}

// ---- records

uint ph(uint h) { return P(PR_HORSES) + h * PH_WORDS; }
uint skill_count(uint h) { return P(ph(h) + PH_SKILL_COUNT); }
uint ps(uint h, uint s) { return P(ph(h) + PH_SKILLS) + s * PS_WORDS; }
int skill_id(uint h, uint s) { return Pi(ps(h, s) + PS_ID); }
uint pd(uint h, uint s, uint d) { return P(ps(h, s) + PS_DETAIL0 + d); }
uint ad(uint h, uint s, uint d) { return P(pd(h, s, d) + PD_STATE); }
uint skill_state(uint h, uint s) { return P(ps(h, s) + PS_STATE); }
int team_id(uint h) { return Pi(ph(h) + PH_TEAM_ID); }

// ---- lists in the state: count, then entries

uint list_count(uint at) { return S(at); }
uint list_get(uint at, uint k) { return S(at + 1u + k); }
bool list_contains(uint at, uint v) {
    for (uint k = 0u; k < S(at); k++) {
        if (S(at + 1u + k) == v) return true;
    }
    return false;
}
void list_push(uint at, uint v) {
    uint n = S(at);
    Sset(at + 1u + n, v);
    Sset(at, n + 1u);
}
void list_remove_at(uint at, uint k) {
    uint n = S(at);
    for (uint j = k; j + 1u < n; j++) Sset(at + 1u + j, S(at + 2u + j));
    Sset(at + n, 0u);
    Sset(at, n - 1u);
}
void list_clear(uint at) {
    for (uint k = 0u; k < S(at); k++) Sset(at + 1u + k, 0u);
    Sset(at, 0u);
}
// A detail in the lists: skill * 2 + detail.
uint detail_code(uint s, uint d) { return s * 2u + d; }


// ---- stopping (mod.rs detail_stop, ability::stop)

void ability_stop(uint pa) {
    uint aa = P(pa + PA_STATE);
    for (uint k = 0u; k < 2u * MAX_RUNS; k++) Sset(aa + AA_RUNS + k, 0u);
    Sset(aa + AA_RUN_COUNT, 0u);
    Sset(aa + AA_ADDITIONAL_ACTIVATE_COUNT, 0u);
}

void detail_stop(uint owner, uint s, uint d) {
    uint det = pd(owner, s, d), a = P(det + PD_STATE);
    if (!Sb(a + AD_IS_ACTIVATED)) return;
    Ssetb(a + AD_IS_ACTIVATED, false);
    for (uint k = 0u; k < P(det + PD_ABILITY_COUNT); k++) {
        uint pa = P(det + PD_ABILITIES) + k * PA_WORDS;
        if (P(pa + PA_PRESENT) != 0u) ability_stop(pa);
    }
    if (P(det + PD_TIME_USAGE) == 4u) {
        Ssetf(a + AD_ADDITIONAL_TIME, 0.0);
        Sseti(a + AD_INCREMENT_COUNT, 0);
    }
    uint acts = P(ph(owner) + PH_ACTIVATE_DETAILS);
    for (uint k = 0u; k < list_count(acts); k++) {
        if (list_get(acts, k) == detail_code(s, d)) {
            list_remove_at(acts, k);
            break;
        }
    }
    Ssetf(a + AD_COOL_DOWN_TIME, Pf(det + PD_DEFAULT_COOL_DOWN_TIME));
}

// skill::clear_skills: a finished horse's details all stop (the skill list is kept).
void clear_skills(uint h) {
    for (uint s = 0u; s < skill_count(h); s++) {
        for (uint d = 0u; d < 2u; d++) detail_stop(h, s, d);
    }
}

// ---- EndOfFrame

// field::cache_prev_activated_skills: each horse's details activated this frame, for the next frame's checks.
// Then field::cache_activate_heal_skill_horse_history_list: in horse order, a horse whose cached details include a
// heal (ability type 9, positive value) moves to the end of the heal history.
void cache_skill_lists() {
    uint n = uint(r_num_horses());
    for (uint h = 0u; h < n; h++) {
        uint prev = P(ph(h) + PH_PREV_DETAILS), cached = P(ph(h) + PH_CACHED_DETAILS);
        list_clear(cached);
        for (uint k = 0u; k < list_count(prev); k++) list_push(cached, list_get(prev, k));
    }
    uint heal = P(PR_HEAL_LIST);
    for (uint h = 0u; h < n; h++) {
        uint cached = P(ph(h) + PH_CACHED_DETAILS);
        bool is_heal = false;
        for (uint k = 0u; k < list_count(cached); k++) {
            uint code = list_get(cached, k);
            uint det = pd(h, code / 2u, code % 2u);
            for (uint ab = 0u; ab < P(det + PD_ABILITY_COUNT); ab++) {
                uint pa = P(det + PD_ABILITIES) + ab * PA_WORDS;
                if (P(pa + PA_PRESENT) != 0u && Pi(pa + PA_TYPE) == 9 && Sf(P(pa + PA_STATE) + AA_VALUE_ON_ACTIVATE) > 0.0) is_heal = true;
            }
        }
        if (!is_heal) continue;
        for (uint k = 0u; k < list_count(heal); k++) {
            if (list_get(heal, k) == h) {
                list_remove_at(heal, k);
                break;
            }
        }
        list_push(heal, h);
    }
}

#endif
