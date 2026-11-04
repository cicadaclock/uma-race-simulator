//! Skill object model and lifecycle: creation, wiz lot, gate check, per-frame update, activation,
//! duration/cooldown and reserved random activation.
//!
//! Addressing: a detail is `(horse i, skill s, detail d)` = `race.horses[i].skill_manager.skills[s].details[d]`.
//! Collections that must run with `&mut Race` (triggers, abilities) can be taken out with `std::mem::take`,
//! updated, and put back — but then code running meanwhile sees the taken slot empty; only do it where
//! nothing else can read that slot.

pub mod ability;
pub mod modifier;
pub mod target;
pub mod trigger;
pub mod value;

use crate::sim::race::Race;
use crate::sim::skill::ability::{SkillAbility};
use crate::sim::skill::trigger::{SkillTrigger, SkillTriggerCollection, TriggerSlot};
use crate::sim::skill::value::{AbilityTime};
use crate::sim::record;
use crate::api::RaceInput;
use crate::data::{GameData, SkillAbilityData, SkillData};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkillDetailRef {
    pub horse: usize,
    pub skill: usize,
    pub detail: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SkillManager {
    /// In input order.
    pub skills: Vec<SkillBase>,
    pub used_skill_ids: Vec<i32>,
    pub prev_activated_details: Vec<SkillDetailRef>,
    pub activate_skills: Vec<SkillDetailRef>,
    pub reserved_skill_ids: Vec<i32>,
    /// Arena holding the triggers of every `SkillTriggerCollection` of this horse's details, contiguous in creation
    /// order (perf: the per-frame trigger walk reads one array per horse).
    pub triggers: Vec<SkillTrigger>,
    /// Set once construction is done. Before that `find_skill` finds nothing, so duplicate ids are not dropped
    /// while skills are added.
    pub has_skill_array: bool,
    /// The observed skills of a horse that has any (see [`crate::api::HorseInput::observed_skills`]). Kept behind one
    /// pointer, so that other horses pay nothing in the per-frame skill walk.
    pub observed: Option<Box<ObservedSkills>>,
    /// Built by the first update after the gate checks.
    pub walk: Option<Box<SkillWalk>>,
}

/// Most skills of one update group that a [`SkillWalk`] holds. A horse with more updates each of them every frame.
const WALK_MAX_SKILLS: usize = 64;

/// Details of one skill.
const WALK_DETAILS: usize = 2;

const WALK_SLOTS: usize = WALK_MAX_SKILLS * WALK_DETAILS;

/// Which skills of a horse could need their per-frame update, by update group (perf: most skills of most horses wait
/// for something). First the group of priority 1, then that of priority 0.
#[derive(Clone, Debug, Default)]
pub struct SkillWalk {
    groups: [SkillWalkGroup; 2],
    /// A group holds more skills than a table can.
    is_oversized: bool,
}

/// One update group, one flat array per value. Each position holds a skill and two slots, one per detail. A slot wakes
/// once the horse has reached both of its thresholds, and a detail that cannot wake holds infinite thresholds. Skills
/// whose next condition has a sentinel term count as awake here, and `is_dormant` then decides.
#[repr(C, align(32))]
#[derive(Clone, Debug)]
struct SkillWalkGroup {
    wake_phase: [f32; WALK_SLOTS],
    wake_distance: [f32; WALK_SLOTS],
    /// Skill index of each position.
    skills: [u16; WALK_MAX_SKILLS],
    /// A bit per position: awake whatever the thresholds say.
    forced: u64,
    count: u32,
}

impl Default for SkillWalkGroup {
    fn default() -> Self {
        SkillWalkGroup {
            wake_phase: [f32::INFINITY; WALK_SLOTS],
            wake_distance: [f32::INFINITY; WALK_SLOTS],
            skills: [0; WALK_MAX_SKILLS],
            forced: 0,
            count: 0,
        }
    }
}

impl SkillWalkGroup {
    /// Stores what the skill at a position waits for.
    fn set_position(&mut self, position: usize, skill: &SkillBase) {
        let mut forced = !(skill.cool_down_time <= 0.0);
        for (d, detail) in skill.details.iter().enumerate() {
            forced |= detail.is_activated | detail.is_use_skill_detail_cool_down_time;
            let next = if detail.is_need_pre_trigger_check { &detail.pre_trigger } else { &detail.trigger };
            let (phase, distance) = match next {
                SkillTriggerCollection::Collection { wake, .. } if skill.is_activate_enable => (wake.phase as f32, wake.distance),
                _ => (f32::INFINITY, f32::INFINITY),
            };
            let slot = position * WALK_DETAILS + d;
            self.wake_phase[slot] = phase;
            self.wake_distance[slot] = distance;
        }
        self.forced = self.forced & !(1u64 << position) | (forced as u64) << position;
    }

    /// A bit per position: the skills that could need their update at this phase and distance.
    fn awake_positions(&self, phase: f32, distance: f32) -> u64 {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the processor has AVX2.
            return unsafe { self.awake_positions_avx2(phase, distance) } | self.forced;
        }
        let mut awake = self.forced;
        for slot in 0..self.count as usize * WALK_DETAILS {
            let reached = (phase >= self.wake_phase[slot]) & (distance >= self.wake_distance[slot]);
            awake |= (reached as u64) << (slot / WALK_DETAILS);
        }
        awake
    }

    /// The reached slots, eight at a time.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn awake_positions_avx2(&self, phase: f32, distance: f32) -> u64 {
        use std::arch::x86_64::*;
        let (phase, distance) = (_mm256_set1_ps(phase), _mm256_set1_ps(distance));
        let mut awake = 0u64;
        let mut slot = 0;
        while slot < self.count as usize * WALK_DETAILS {
            // SAFETY: `slot + 8` stays within the arrays, because WALK_SLOTS is a multiple of 8, and the group's alignment
            // is that of a vector.
            let (wake_phase, wake_distance) =
                unsafe { (_mm256_load_ps(self.wake_phase.as_ptr().add(slot)), _mm256_load_ps(self.wake_distance.as_ptr().add(slot))) };
            let reached = _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(phase, wake_phase), _mm256_cmp_ps::<_CMP_GE_OQ>(distance, wake_distance));
            let slots = _mm256_movemask_ps(reached) as u64;
            // Two bits per skill to one
            let either = (slots | slots >> 1) & 0x55;
            let skills = (either | either >> 1) & 0x33;
            awake |= ((skills | skills >> 2) & 0x0F) << (slot / WALK_DETAILS);
            slot += 8;
        }
        awake
    }
}

impl SkillWalk {
    /// The table of a horse's skills as they stand.
    fn new(skills: &[SkillBase]) -> Box<SkillWalk> {
        let mut walk = Box::<SkillWalk>::default();
        for (s, skill) in skills.iter().enumerate() {
            let Some(group) = [1, 0].iter().position(|&priority| priority == skill.priority) else { continue };
            let table = &mut walk.groups[group];
            let position = table.count as usize;
            if position == WALK_MAX_SKILLS {
                walk.is_oversized = true;
                return walk;
            }
            table.skills[position] = s as u16;
            table.count += 1;
            table.set_position(position, skill);
        }
        walk
    }
}

/// What a horse holds for its observed skills, indexed by skill.
#[derive(Clone, Debug, Default)]
pub struct ObservedSkills {
    /// The own generator of an observed skill.
    pub own_rngs: Vec<Option<Box<crate::rng::RaceRandom>>>,
    /// Whether the skill is observed and muted, so that everything else in the race treats it as absent.
    pub silent: Vec<bool>,
}

impl SkillManager {
    /// Whether skill index `s` is silent.
    pub fn is_silent(&self, s: usize) -> bool {
        self.observed.as_ref().is_some_and(|o| o.silent.get(s).is_some_and(|&silent| silent))
    }

    /// Whether skill index `s` is observed.
    fn is_observed(&self, s: usize) -> bool {
        self.observed.as_ref().is_some_and(|o| o.own_rngs.get(s).is_some_and(Option::is_some))
    }

    /// Adds observed skill `s` of `source` in the state that it holds there, as a skill that is not silent. With this
    /// manager at the same point of the race without the skill, and no effect of the skill so far, the race continues
    /// as if the skill had been held from the start, given [`stays_baseline_until_first_effect`].
    pub fn adopt_observed_skill(&mut self, source: &SkillManager, s: usize) {
        let mut skill = source.skills[s].clone();
        let ranges = skill.details.iter().flat_map(|d| [d.pre_trigger.range(), d.trigger.range()]).filter(|r| !r.is_empty());
        let first = ranges.clone().map(|r| r.start).min().unwrap_or(0);
        let last = ranges.map(|r| r.end).max().unwrap_or(0);
        let offset = self.triggers.len() as i64 - first as i64;
        self.triggers.extend_from_slice(&source.triggers[first..last]);
        for d in &mut skill.details {
            d.pre_trigger.move_in_arena(offset);
            d.trigger.move_in_arena(offset);
        }
        let index = self.skills.len();
        self.skills.push(skill);
        let own = source.observed.as_ref().and_then(|o| o.own_rngs[s].clone());
        let observed = self.observed.get_or_insert_default();
        observed.own_rngs.resize_with(index, || None);
        observed.own_rngs.push(own);
        observed.silent.resize(index + 1, false);
    }
}

/// Whether the race stays bit for bit the race without skill `id` until its first effect, when horse `owner` holds it
/// as an observed skill that is not silent. It does not when the skill sets the running style at the start, when a
/// skill of the owner scales with the owner's skill count, or when another horse holds the skill and counts its holders.
pub fn stays_baseline_until_first_effect(data: &GameData, input: &RaceInput, owner: usize, id: i32) -> bool {
    let Some(m) = data.skills.get(&id) else { return false };
    let sets_running_style = skill_abilities(m).any(|a| a.ability_type == 6);
    let scales_with_skill_count = input.horses[owner]
        .skills
        .iter()
        .filter_map(|s| data.skills.get(&s.skill_id))
        .any(|s| skill_abilities(s).any(|a| a.ability_value_usage == 2));
    let counts = |text: &str| text.contains("same_skill_horse_count");
    let counts_holders = m.details.iter().any(|d| counts(&d.condition) || counts(&d.precondition));
    let other_holder = input.horses.iter().enumerate().any(|(j, h)| j != owner && h.skills.iter().any(|s| s.skill_id == id));
    !(sets_running_style || scales_with_skill_count || (counts_holders && other_holder))
}

fn skill_abilities(skill: &SkillData) -> impl Iterator<Item = &SkillAbilityData> {
    skill.details.iter().flat_map(|d| &d.abilities)
}

#[derive(Clone, Debug, Default)]
pub struct SkillBase {
    pub is_activate_enable: bool,
    /// Detail 0 = the `*_1` master columns, detail 1 = `*_2`.
    pub details: [SkillDetail; 2],
    /// Era-effective master row.
    pub skill_master: Box<SkillData>,
    /// Hot copies of `skill_master.id` / `.priority` (perf: the per-frame loops read only these).
    pub id: i32,
    pub priority: i32,
    pub level: i32,
    pub cool_down_time: f32,
}

/// One condition/effect half of a skill.
#[derive(Clone, Debug, Default)]
pub struct SkillDetail {
    pub detail_index: i32,
    pub trigger: SkillTriggerCollection,
    pub pre_trigger: SkillTriggerCollection,
    pub is_need_pre_trigger_check: bool,
    pub default_cool_down_time: f32,
    pub cool_down_time: f32,
    pub is_use_skill_detail_cool_down_time: bool,
    pub is_not_activate_skill: bool,
    pub is_time_infinite: bool,
    pub left_time: f32,
    pub time: AbilityTime,
    pub skill_level: i32,
    pub is_debuff: bool,
    /// Speed 0, Heal 1, Accel 2, none -1.
    pub category: i32,
    /// `None` = an ability type without effect (6, 501..503, unknown).
    pub abilities: Vec<Option<SkillAbility>>,
    pub is_activated: bool,
}

pub fn detail<'r>(race: &'r Race, r: SkillDetailRef) -> &'r SkillDetail {
    &race.horses[r.horse].skill_manager.skills[r.skill].details[r.detail]
}

pub fn detail_mut<'r>(race: &'r mut Race, r: SkillDetailRef) -> &'r mut SkillDetail {
    &mut race.horses[r.horse].skill_manager.skills[r.skill].details[r.detail]
}

pub fn skill_id(race: &Race, r: SkillDetailRef) -> i32 {
    race.horses[r.horse].skill_manager.skills[r.skill].id
}

/// Never finds anything during construction, see `has_skill_array`.
pub fn find_skill(sm: &SkillManager, id: i32) -> Option<usize> {
    if !sm.has_skill_array {
        return None;
    }
    sm.skills.iter().position(|s| s.id == id)
}

pub fn tag_ids(m: &SkillData) -> Vec<i32> {
    m.tag_id.split('/').filter_map(|t| t.trim().parse().ok()).collect()
}

/// Whether an ability with this type and value is a debuff.
pub fn is_skill_demerit(ability_type: i32, v: f32) -> bool {
    match ability_type {
        1..=5 | 8 | 9 | 21 | 22 | 27 | 28 | 30..=32 => v < 0.0,
        10 => v > 1.0,
        13 | 14 => true,
        29 => !(v < 0.0),
        _ => false,
    }
}

/// Speed 0, Heal 1, Accel 2, none -1.
pub fn skill_category(ability_type: i32) -> i32 {
    match ability_type {
        7 | 9 | 13 => 1,
        21 | 22 | 27 => 0,
        31 => 2,
        _ => -1,
    }
}

/// Era-effective `skill_data` row; `None` for an unknown id (the skill is skipped).
pub fn lookup_skill_master(race: &Race, skill_id: i32) -> Option<SkillData> {
    race.data.skills.get(&skill_id).cloned()
}

/// Trigger construction may draw RNG, in order: detail 1 pre, cond, abilities 1..3, then detail 2.
pub fn create_skill(race: &mut Race, i: usize) {
    race.horses[i].skill_manager = SkillManager::default();
    let list = race.horses[i].horse_data.skills.clone();
    for s in list {
        let Some(m) = lookup_skill_master(race, s.skill_id) else { continue };
        // Duplicate check (never true here, see `has_skill_array`).
        if find_skill(&race.horses[i].skill_manager, m.id).is_some() {
            continue;
        }
        let id = m.id;
        let sm = &mut race.horses[i].skill_manager;
        sm.skills.push(SkillBase { id, priority: m.priority, skill_master: Box::new(m), level: s.level, ..Default::default() });
        let si = sm.skills.len() - 1;
        if race.input.horses[i].observed_skills.contains(&id) {
            let mixed = (race.info.random_seed as u32).wrapping_mul(0x9E37_79B1) ^ (id as u32).wrapping_mul(0x85EB_CA77);
            let silent = race.input.horses[i].is_silent_skill(id);
            let observed = race.horses[i].skill_manager.observed.get_or_insert_default();
            observed.own_rngs.resize_with(si + 1, || None);
            observed.silent.resize(si + 1, false);
            observed.own_rngs[si] = Some(Box::new(crate::rng::RaceRandom::new((mixed & 0x7fff_ffff) as i32)));
            observed.silent[si] = silent;
        }
        on_skill_stream(race, i, si, |race| setup_detail(race, i, si));
    }
    race.horses[i].skill_manager.has_skill_array = true;
    race.horses[i].is_skill_initialized = true;
}

/// Whether skill `id` can never activate for horse `owner` in this race: every detail with conditions has a
/// precondition or condition that is [`trigger::never_true`]. Unknown skills and ones without conditions count as
/// possible. May draw RNG (see there).
pub fn never_activates(race: &mut Race, owner: usize, id: i32) -> bool {
    let data = race.data;
    let Some(s) = data.skills.get(&id) else { return false };
    let mut used = s.details.iter().filter(|d| !d.precondition.is_empty() || !d.condition.is_empty()).peekable();
    used.peek().is_some() && used.all(|d| trigger::never_true(race, owner, &d.precondition) || trigger::never_true(race, owner, &d.condition))
}

fn setup_detail(race: &mut Race, i: usize, si: usize) {
    let course_distance = race.info.race_course_set.distance;
    let (div_time, div_cd) = (race.data.param.Skill.AbilityTimeDivideDistance, race.data.param.Skill.CoolDownTimeDivideDistance);
    for d in 0..2 {
        let base = &race.horses[i].skill_manager.skills[si];
        let level = base.level;
        let dd = base.skill_master.details[d].clone();
        let r = SkillDetailRef { horse: i, skill: si, detail: d };
        let ability_time = dd.float_ability_time as f32 * 0.0001;
        let cool_down = dd.float_cooldown_time as f32 * 0.0001;
        let dist_rate_time = value::calc_ability_time_by_distance_rate(course_distance, div_time);
        let ability_time = dist_rate_time * ability_time;
        let tc = value::AbilityTime::new(dd.ability_time_usage, ability_time, dist_rate_time);
        race.horses[i].skill_manager.skills[si].details[d] = SkillDetail {
            detail_index: d as i32,
            time: tc,
            cool_down_time: 0.0,
            is_time_infinite: ability_time < 0.0,
            default_cool_down_time: value::calc_cool_down_time_by_distance(cool_down, course_distance, div_cd),
            skill_level: level,
            ..Default::default()
        };
        // Trigger construction may draw RNG.
        let pre = trigger::create_trigger_collection(race, r, &dd.precondition);
        let cond = trigger::create_trigger_collection(race, r, &dd.condition);
        let arena = &race.horses[i].skill_manager.triggers;
        let other_detail = trigger::collection_has_activate_other_skill_detail(arena, &pre)
            || trigger::collection_has_activate_other_skill_detail(arena, &cond);
        let det = detail_mut(race, r);
        det.is_need_pre_trigger_check = !matches!(pre, SkillTriggerCollection::Null);
        det.is_use_skill_detail_cool_down_time = other_detail;
        det.is_not_activate_skill = other_detail;
        det.pre_trigger = pre;
        det.trigger = cond;
        let a1 = &dd.abilities[0];
        det.is_debuff = a1.ability_type != 0 && is_skill_demerit(a1.ability_type, a1.float_ability_value as f32 * 0.0001);
        det.category = skill_category(a1.ability_type);
        for ab in &dd.abilities {
            if ab.ability_type != 0 {
                if let Some(a) = create_ability(race, r, level, ab) {
                    detail_mut(race, r).abilities.push(a);
                }
            }
        }
    }
}

/// `None` = not added; `Some(None)` = an ability without effect. The target is always created.
fn create_ability(race: &mut Race, r: SkillDetailRef, level: i32, ab: &SkillAbilityData) -> Option<Option<SkillAbility>> {
    let t = ab.ability_type;
    let value = ab.float_ability_value as f32 * 0.0001;
    let is_demerit = t != 0 && is_skill_demerit(t, value);
    let target = target::SkillTarget::new(r.horse, ab.target_type, ab.target_value, is_demerit);
    if t < 1 {
        return None;
    }
    match t {
        1..=5 | 7..=14 | 21 | 22 | 27..=32 | 35..=38 => Some(Some(ability::SkillAbility::new(race, r, level, ab, target))),
        _ => Some(None),
    }
}

/// Runs `body` with the generator of an observed skill in place of the race's, and plainly for any other skill.
fn on_skill_stream<R>(race: &mut Race, i: usize, s: usize, body: impl FnOnce(&mut Race) -> R) -> R {
    let own = race.horses[i].skill_manager.observed.as_mut().and_then(|o| o.own_rngs.get_mut(s)).and_then(Option::take);
    let Some(mut own) = own else { return body(race) };
    std::mem::swap(&mut race.rng, &mut *own);
    let result = body(race);
    std::mem::swap(&mut race.rng, &mut *own);
    race.horses[i].skill_manager.observed.as_mut().expect("the skill was observed").own_rngs[s] = Some(own);
    result
}

/// Pre-race wiz lot: decides which `activate_lot` skills may activate this race. `base_wiz` is the halved wiz.
pub fn lot_activate_skill(race: &mut Race, i: usize) {
    let p = &race.data.param.Skill;
    let (v1, v2, min) = (p.LotActivatePerVal1, p.LotActivatePerVal2, p.ActivatePerMin as f32);
    for s in 0..race.horses[i].skill_manager.skills.len() {
        let enable = if race.horses[i].skill_manager.skills[s].skill_master.activate_lot == 1 {
            let wiz = crate::sim::physics::stats::base_wiz_for(&race.horses[i], crate::api::wiz_mechanic::SKILLS);
            // One f64 expression, rounded at the store.
            let thr = (v1 as f64 - v2 as f64 / wiz as f64) as f32;
            let thr = if min > thr { min } else { thr };
            let r = on_skill_stream(race, i, s, |race| race.rng.random_float(100.0));
            r < thr
        } else {
            true
        };
        race.horses[i].skill_manager.skills[s].is_activate_enable = enable;
    }
}

pub fn check_skill_trigger_and_activate_first_group(race: &mut Race, i: usize) {
    race.horses[i].skill_manager.walk = None;
    for s in 0..race.horses[i].skill_manager.skills.len() {
        if race.horses[i].skill_manager.skills[s].priority == 1 {
            on_skill_stream(race, i, s, |race| skill_base_check_trigger_and_activate(race, i, s));
        }
    }
}

pub fn check_skill_trigger_and_activate_second_group(race: &mut Race, i: usize) {
    race.horses[i].skill_manager.walk = None;
    for s in 0..race.horses[i].skill_manager.skills.len() {
        if race.horses[i].skill_manager.skills[s].priority == 0 {
            on_skill_stream(race, i, s, |race| skill_base_check_trigger_and_activate(race, i, s));
        }
    }
}

/// Gate check of one skill (dt = 0): no cooldown, no break, already-activated details are re-checked.
pub fn skill_base_check_trigger_and_activate(race: &mut Race, i: usize, s: usize) {
    let enable = race.horses[i].skill_manager.skills[s].is_activate_enable;
    for d in 0..race.horses[i].skill_manager.skills[s].details.len() {
        let r = SkillDetailRef { horse: i, skill: s, detail: d };
        if (enable || detail_is_reserved_random_activate(race, r)) && detail_update_trigger_and_activate(race, r, 0.0) {
            add_prev_activate_skill_detail(race, r);
        }
    }
}

pub fn update_first_group(race: &mut Race, i: usize, dt: f32) {
    if race.horses[i].skill_manager.observed.is_some() {
        return update_group_with_observed_skills(race, i, 1, dt);
    }
    update_walked_group(race, i, 0, 1, dt);
}

pub fn update_second_group(race: &mut Race, i: usize, dt: f32) {
    if race.horses[i].skill_manager.observed.is_some() {
        update_group_with_observed_skills(race, i, 0, dt);
    } else {
        update_walked_group(race, i, 1, 0, dt);
    }
    update_random_activate_skill_id_list(race, i);
}

/// One update group of a horse without observed skills, through its table. Skipping a skill that is certainly dormant
/// changes nothing, and only a skill's own update changes what that skill waits for. A skill's activation can make the
/// later skills of the frame busy, so that test runs before each of them.
fn update_walked_group(race: &mut Race, i: usize, group: usize, priority: i32, dt: f32) {
    let h = &mut race.horses[i];
    let (phase, distance) = (h.phase as i32 as f32, h.distance);
    let sm = &mut h.skill_manager;
    let walk = sm.walk.get_or_insert_with(|| SkillWalk::new(&sm.skills));
    if walk.is_oversized {
        for s in 0..race.horses[i].skill_manager.skills.len() {
            if race.horses[i].skill_manager.skills[s].priority == priority {
                skill_base_update(race, i, s, dt);
            }
        }
        return;
    }
    let table = &walk.groups[group];
    let (count, awake) = (table.count as usize, table.awake_positions(phase, distance));
    let mut position = 0;
    while position < count {
        let sm = &race.horses[i].skill_manager;
        if sm.prev_activated_details.is_empty() && sm.reserved_skill_ids.is_empty() {
            let rest = awake >> position;
            if rest == 0 {
                return;
            }
            position += rest.trailing_zeros() as usize;
        }
        let s = sm.walk.as_ref().expect("the walk was built above").groups[group].skills[position] as usize;
        if !is_dormant(race, i, s) {
            update_awake_skill(race, i, s, dt);
            let sm = &mut race.horses[i].skill_manager;
            sm.walk.as_mut().expect("the walk was built above").groups[group].set_position(position, &sm.skills[s]);
        }
        position += 1;
    }
}

/// One update group of a horse with observed skills, each observed skill on its generator. A dormant observed skill
/// keeps its generator in place, because the swap would cost more than the update it skips.
#[cold]
fn update_group_with_observed_skills(race: &mut Race, i: usize, priority: i32, dt: f32) {
    for s in 0..race.horses[i].skill_manager.skills.len() {
        if race.horses[i].skill_manager.skills[s].priority != priority {
            continue;
        }
        if !race.horses[i].skill_manager.is_observed(s) {
            skill_base_update(race, i, s, dt);
        } else if !is_dormant(race, i, s) {
            on_skill_stream(race, i, s, |race| skill_base_update(race, i, s, dt));
        }
    }
}

/// The skill list itself is kept; finished horses no longer update skills.
pub fn clear_skills(race: &mut Race, i: usize) {
    for s in 0..race.horses[i].skill_manager.skills.len() {
        for d in 0..race.horses[i].skill_manager.skills[s].details.len() {
            detail_stop(race, SkillDetailRef { horse: i, skill: s, detail: d });
        }
    }
}

fn add_prev_activate_skill_detail(race: &mut Race, r: SkillDetailRef) {
    if race.horses[r.horse].skill_manager.is_silent(r.skill) {
        return;
    }
    let l = &mut race.horses[r.horse].skill_manager.prev_activated_details;
    if !l.contains(&r) {
        l.push(r);
    }
}

/// Per-frame update of one skill: active details run; otherwise the skill cooldown ticks and the first detail whose
/// trigger fires activates. Details with their own cooldown are handled separately at the end.
pub fn skill_base_update(race: &mut Race, i: usize, s: usize, dt: f32) {
    if !is_dormant(race, i, s) {
        update_awake_skill(race, i, s, dt);
    }
}

/// `skill_base_update` of a skill that is not dormant.
fn update_awake_skill(race: &mut Race, i: usize, s: usize, dt: f32) {
    let nd = race.horses[i].skill_manager.skills[s].details.len();
    let rf = |d: usize| SkillDetailRef { horse: i, skill: s, detail: d };
    if race.horses[i].is_complete_first_update && !race.horses[i].skill_manager.prev_activated_details.is_empty() {
        for d in 0..nd {
            let l = &mut race.horses[i].skill_manager.prev_activated_details;
            if let Some(k) = l.iter().position(|x| *x == rf(d)) {
                l.remove(k);
            }
        }
    }
    let any_normal_active = (0..nd).any(|d| {
        let det = detail(race, rf(d));
        !det.is_use_skill_detail_cool_down_time && det.is_activated
    });
    if any_normal_active {
        for d in 0..nd {
            let det = detail(race, rf(d));
            if det.is_activated && !det.is_use_skill_detail_cool_down_time && detail_update_skill(race, rf(d), dt) {
                let cd = detail(race, rf(d)).default_cool_down_time;
                race.horses[i].skill_manager.skills[s].cool_down_time = cd;
            }
        }
    } else {
        let base = &mut race.horses[i].skill_manager.skills[s];
        let enable = base.is_activate_enable;
        if 0.0 < base.cool_down_time {
            base.cool_down_time -= dt;
            if base.cool_down_time < 0.0 {
                base.cool_down_time = 0.0;
            }
        }
        if base.cool_down_time <= 0.0 {
            for d in 0..nd {
                if !(enable || detail_is_reserved_random_activate(race, rf(d))) {
                    continue;
                }
                let det = detail(race, rf(d));
                if !det.is_activated && !det.is_use_skill_detail_cool_down_time && detail_update_trigger_and_activate(race, rf(d), dt) {
                    add_prev_activate_skill_detail(race, rf(d));
                    break;
                }
            }
        }
    }
    for d in 0..nd {
        let det = detail(race, rf(d));
        if !det.is_use_skill_detail_cool_down_time {
            continue;
        }
        if !det.is_activated {
            if detail_check_cool_down(race, rf(d), dt)
                && detail_update_trigger_and_activate(race, rf(d), dt)
                && !detail(race, rf(d)).is_not_activate_skill
            {
                add_prev_activate_skill_detail(race, rf(d));
            }
        } else {
            detail_update_skill(race, rf(d), dt);
        }
    }
}

/// Whether `skill_base_update` would change nothing: idle, no cooldown running, and every detail's next trigger
/// check (the precondition while it is pending) is asleep (see [`trigger::Wake`]) or empty.
fn is_dormant(race: &Race, i: usize, s: usize) -> bool {
    let h = &race.horses[i];
    let sm = &h.skill_manager;
    let b = &sm.skills[s];
    if !sm.prev_activated_details.is_empty() || !sm.reserved_skill_ids.is_empty() || !(b.cool_down_time <= 0.0) {
        return false;
    }
    b.details.iter().all(|d| {
        let next = if d.is_need_pre_trigger_check { &d.pre_trigger } else { &d.trigger };
        !d.is_activated
            && !d.is_use_skill_detail_cool_down_time
            && (!b.is_activate_enable
                || match next {
                    SkillTriggerCollection::Null => true,
                    SkillTriggerCollection::Collection { wake, .. } => wake.is_asleep(race, i),
                })
    })
}

/// Ticks the detail's own cooldown; true once it is over.
fn detail_check_cool_down(race: &mut Race, r: SkillDetailRef, dt: f32) -> bool {
    let det = detail_mut(race, r);
    if 0.0 < det.cool_down_time {
        det.cool_down_time -= dt;
        if det.cool_down_time < 0.0 {
            det.cool_down_time = 0.0;
        }
    }
    det.cool_down_time <= 0.0
}

/// Checks the triggers and activates the detail. The precondition latches: once met it is not checked again.
pub fn detail_update_trigger_and_activate(race: &mut Race, r: SkillDetailRef, dt: f32) -> bool {
    let reserved = detail_is_reserved_random_activate(race, r);
    let det = detail(race, r);
    let check_pre = !reserved && det.is_need_pre_trigger_check;
    // Nothing to update (a detail without conditions, e.g. an unused second detail).
    if !check_pre && matches!(det.trigger, SkillTriggerCollection::Null) {
        return false;
    }
    if check_pre {
        if !trigger::collection_update(race, r, TriggerSlot::Pre, dt, reserved) {
            return false;
        }
        detail_mut(race, r).is_need_pre_trigger_check = false;
    }
    if !trigger::collection_update(race, r, TriggerSlot::Main, dt, reserved) {
        return false;
    }
    detail_activate(race, r);
    true
}

/// Bit per target horse index of the first ability (none for an ability without effect).
fn calc_target_flags(race: &Race, r: SkillDetailRef) -> i32 {
    let Some(Some(ab)) = detail(race, r).abilities.first() else { return 0 };
    ab.target.targets.iter().fold(0u32, |f, &h| f | 1u32 << (race.horses[h].horse_data.horse_index & 31)) as i32
}

pub fn detail_activate(race: &mut Race, r: SkillDetailRef) {
    detail_mut(race, r).is_activated = true;
    let t = value::ability_time(race, r);
    detail_mut(race, r).left_time = t;
    for k in 0..detail(race, r).abilities.len() {
        if detail(race, r).abilities[k].is_some() {
            ability::activate(race, r, k, false);
        }
    }
    record_skill_event(race, r);
    let det = detail(race, r);
    if det.is_not_activate_skill {
        return;
    }
    let i = r.horse;
    let m = &race.horses[i].skill_manager.skills[r.skill].skill_master;
    let id = m.id;
    let tags = tag_ids(m);
    let is_heal = det.abilities.iter().flatten().any(|ab| ab.ability_type == 9 && ab.ability_value_on_activate > 0.0);
    let course_distance = race.info.race_course_set.distance;
    let tag_group = &race.data.param.Skill.AbilityValueUsageTagGroup1Array;
    let specific = !tag_group.is_empty() && tags.iter().any(|t| tag_group.contains(t));
    let h = &mut race.horses[i];
    if !h.skill_manager.activate_skills.contains(&r) {
        h.skill_manager.activate_skills.push(r);
    }
    if h.skill_manager.is_silent(r.skill) {
        let infinite = detail(race, r).is_time_infinite;
        record::open_skill_span(race, r, id, infinite);
        return;
    }
    if !h.skill_manager.used_skill_ids.contains(&id) {
        h.skill_manager.used_skill_ids.push(id);
    }
    // Counted only for Start..LastSpurt.
    let phase = h.phase as usize;
    if phase < h.activate_skill_count_by_phase.len() {
        h.activate_skill_count_by_phase[phase] += 1;
    }
    let second_half = course_distance as f32 * 0.5 <= h.distance;
    h.activate_skill_count_by_distance[second_half as usize] += 1;
    if is_heal {
        h.activate_heal_skill_count += 1;
    }
    if specific {
        h.activate_specific_tag_group_skill_count += 1;
    }
    let infinite = detail(race, r).is_time_infinite;
    record::open_skill_span(race, r, id, infinite);
}

/// At the gate (time <= 0) the target flags are 0 and the continue time is -1.
fn record_skill_event(race: &mut Race, r: SkillDetailRef) {
    let time = race.accumulate_time_since_start;
    let det = detail(race, r);
    let (flags, continue_time) = if 0.0 < time {
        // The continue time is computed in f64.
        (calc_target_flags(race, r), crate::sim::util::f64_to_i32(det.left_time as f64 * 10000.0))
    } else {
        (0, -1)
    };
    let (detail_index, activate_type) = (det.detail_index, det.is_not_activate_skill as i32);
    let id = skill_id(race, r);
    let horse_index = race.horses[r.horse].horse_data.horse_index as usize;
    record::add_skill_event(race, horse_index, id, detail_index, flags, time, continue_time, activate_type);
}

/// Returns true when the detail ended this frame.
pub fn detail_update_skill(race: &mut Race, r: SkillDetailRef, dt: f32) -> bool {
    if detail(race, r).is_activated {
        for k in 0..detail(race, r).abilities.len() {
            if detail(race, r).abilities[k].is_none() {
                continue;
            }
            let n = ability::calc_additional_activate_count(race, r, k);
            for _ in 0..n {
                ability::activate(race, r, k, true);
            }
        }
        if !detail(race, r).is_time_infinite {
            value::update_ability_time(race, r);
            let add = value::additional_ability_time(race, r);
            if 0.0 < add {
                let det = detail_mut(race, r);
                det.left_time = add + det.left_time;
                for k in 0..detail(race, r).abilities.len() {
                    if detail(race, r).abilities[k].is_some() {
                        ability::add_ability_time(race, r, k, add);
                    }
                }
            }
        }
    }
    let det = detail_mut(race, r);
    if det.is_time_infinite {
        return false;
    }
    det.left_time -= dt;
    if 0.0 < det.left_time {
        return false;
    }
    detail_stop(race, r);
    true
}

pub fn detail_stop(race: &mut Race, r: SkillDetailRef) {
    if !detail(race, r).is_activated {
        return;
    }
    record::close_skill_span(race, r);
    detail_mut(race, r).is_activated = false;
    for k in 0..detail(race, r).abilities.len() {
        if detail(race, r).abilities[k].is_some() {
            ability::stop(race, r, k);
        }
    }
    value::time_calculator_init(race, r);
    let l = &mut race.horses[r.horse].skill_manager.activate_skills;
    if let Some(k) = l.iter().position(|x| *x == r) {
        l.remove(k);
    }
    let det = detail_mut(race, r);
    det.cool_down_time = det.default_cool_down_time;
}

/// Whether this is detail 0 of a skill reserved for random activation.
pub fn detail_is_reserved_random_activate(race: &Race, r: SkillDetailRef) -> bool {
    r.detail == 0 && race.horses[r.horse].skill_manager.reserved_skill_ids.contains(&skill_id(race, r))
}

/// Re-draws the skills reserved for random activation (ability types 36/37, which 564 Escapades holds): per modifier,
/// up to n eligible skills drawn from the race RNG.
pub fn update_random_activate_skill_id_list(race: &mut Race, i: usize) {
    use crate::sim::defs::SkillModifierParam as P;
    race.horses[i].skill_manager.reserved_skill_ids.clear();
    for (param, rarities) in [(P::ActivateRandomRareSkill, &[2, 6][..]), (P::ActivateRandomNormalAndRareSkill, &[1, 2, 6][..])] {
        let n = crate::sim::util::f32_to_i32(race.horses[i].skill_modifier_receiver.apply_modifier(param, 0.0));
        if n <= 0 {
            continue;
        }
        let sm = &race.horses[i].skill_manager;
        let eligible = |s: &SkillBase| {
            let m = &s.skill_master;
            rarities.contains(&m.rarity)
                && m.details[0].float_ability_time != -1
                && s.cool_down_time <= 0.0
                && m.is_general_skill == 1
                && sm.activate_skills.iter().all(|d| race.horses[d.horse].skill_manager.skills[d.skill].skill_master.id != m.id)
                && !sm.reserved_skill_ids.contains(&m.id)
        };
        let skills = sm.skills.iter().enumerate();
        let mut cand: Vec<i32> = skills.clone().filter(|(k, s)| !sm.is_observed(*k) && eligible(s)).map(|(_, s)| s.skill_master.id).collect();
        let observed: Vec<usize> = skills.filter(|(k, s)| sm.is_observed(*k) && eligible(s)).map(|(k, _)| k).collect();
        let pool = cand.len() as i32 + 1;
        let drawn = n.min(cand.len() as i32);
        for _ in 0..drawn {
            let k = race.rng.random.next_range(0, cand.len() as i32) as usize;
            race.horses[i].skill_manager.reserved_skill_ids.push(cand.remove(k));
        }
        // An observed skill stays out of the draw so that holding it moves no other pick. It takes a pick on its own
        // generator with the chance that it would have in a pool that held it; a silent one takes nothing from the race.
        for k in observed {
            let sm = &mut race.horses[i].skill_manager;
            let (id, silent) = (sm.skills[k].id, sm.is_silent(k));
            let own = sm.observed.as_mut().and_then(|o| o.own_rngs[k].as_mut()).expect("an observed skill holds a generator");
            if own.random.next_range(0, pool) < n.min(pool) {
                if !silent && drawn > 0 {
                    sm.reserved_skill_ids.pop();
                }
                sm.reserved_skill_ids.push(id);
            }
        }
    }
}
