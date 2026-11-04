//! Skill conditions: a condition string parses into a collection of triggers (OR-groups of AND-terms) that
//! updates every frame; plus the per-horse order-in/out-continue flags the order-rate triggers read.
//!
//! Every trigger is a [`SkillTrigger`] (shared fields) + a [`SkillTriggerKind`] (the keyword's own state).

pub mod kind;
pub mod order_in_out;
pub mod read;
pub mod window;

use crate::sim::ai;
use crate::sim::race::Race;
use crate::sim::skill::trigger::kind::{create_trigger, SkillTriggerKind};
use crate::sim::field::{first_horse, horse_by_order, last_horse_index};
use crate::sim::skill::trigger::read::{activate_skill_count, value_lh};
use crate::sim::skill::trigger::window::{all_corner_random_init, calc_final_corner_distance, init_distance};
use crate::sim::skill::{self, SkillDetailRef};
use crate::sim::util::{approximately, f32_to_i32};

#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SkillTriggerOperator {
    #[default]
    Equal = 0,
    NotEqual = 1,
    Greater = 2,
    GreaterEqual = 3,
    Lesser = 4,
    LesserEqual = 5,
}

/// A parsed condition string; `Null` for an empty one.
#[derive(Clone, Debug, Default)]
pub enum SkillTriggerCollection {
    #[default]
    Null,
    /// OR-groups (split by `@`) of AND-terms (split by `&`), left to right.
    /// Flattened (perf): the triggers live contiguously in the owner's [`skill::SkillManager::triggers`] arena at
    /// `start..start + len`, in the same left-to-right order, each tagged with its group index
    /// ([`SkillTrigger::group`]); an empty group (`"a@"`) has no triggers, so it is kept as `any_empty_group`
    /// (an empty AND-group counts as activated). `has_lot_other`: some term is random_lot_other_activate.
    Collection { start: u32, len: u32, any_empty_group: bool, has_lot_other: bool, wake: Wake },
}

/// When a collection of pure terms is certainly false, so it need not be updated (perf: most conditions wait for
/// something): below `phase` or `distance` (both only grow), or while its one group's `sentinel` term (an arena
/// index; `NO_SENTINEL` for none) is false.
#[derive(Clone, Copy, Debug)]
pub struct Wake {
    pub phase: i32,
    pub distance: f32,
    pub sentinel: u32,
}

const NO_SENTINEL: u32 = u32::MAX;

impl Wake {
    const NOW: Wake = Wake { phase: i32::MIN, distance: f32::NEG_INFINITY, sentinel: NO_SENTINEL };
    const NEVER: Wake = Wake { phase: i32::MAX, distance: f32::INFINITY, sentinel: NO_SENTINEL };

    fn phase(phase: i32) -> Wake {
        Wake { phase, ..Wake::NOW }
    }

    fn distance(distance: f32) -> Wake {
        Wake { distance, ..Wake::NOW }
    }

    /// Every term must be true: the later threshold of each.
    fn and(self, o: Wake) -> Wake {
        Wake { phase: self.phase.max(o.phase), distance: self.distance.max(o.distance), sentinel: NO_SENTINEL }
    }

    /// Some group must be true: the earlier threshold of each (conservative).
    fn or(self, o: Wake) -> Wake {
        Wake { phase: self.phase.min(o.phase), distance: self.distance.min(o.distance), sentinel: NO_SENTINEL }
    }

    /// Whether the collection is certainly false now, for owner `owner`.
    pub fn is_asleep(self, race: &Race, owner: usize) -> bool {
        let h = &race.horses[owner];
        (h.phase as i32) < self.phase
            || h.distance < self.distance
            || (self.sentinel != NO_SENTINEL && !check_pure(race, owner, &h.skill_manager.triggers[self.sentinel as usize]))
    }
}

impl Default for Wake {
    fn default() -> Self {
        Wake::NOW
    }
}

impl SkillTriggerCollection {
    /// This collection's triggers in the owner's `SkillManager::triggers` arena.
    pub fn range(&self) -> std::ops::Range<usize> {
        match *self {
            SkillTriggerCollection::Null => 0..0,
            SkillTriggerCollection::Collection { start, len, .. } => start as usize..(start + len) as usize,
        }
    }
}

/// One condition term: `keyword op value_rh`. `kind == Null` (unknown keyword) is never activated and never updates.
#[derive(Clone, Debug, Default)]
pub struct SkillTrigger {
    pub op: SkillTriggerOperator,
    pub value_rh: i32,
    pub is_activated: bool,
    /// Index of the OR-group in the collection (see [`SkillTriggerCollection`]).
    pub group: u16,
    pub kind: SkillTriggerKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriggerSlot {
    /// precondition_N.
    Pre,
    /// condition_N.
    Main,
}

/// Splits a term into (keyword, operator, value). Exactly one operator; the value is an integer with optional
/// surrounding white space and sign. Two-char operators are tried first.
pub fn parse_term(term: &str) -> Option<(&str, SkillTriggerOperator, i32)> {
    const OPS: [(&str, SkillTriggerOperator); 6] = [
        ("==", SkillTriggerOperator::Equal),
        ("!=", SkillTriggerOperator::NotEqual),
        (">=", SkillTriggerOperator::GreaterEqual),
        ("<=", SkillTriggerOperator::LesserEqual),
        (">", SkillTriggerOperator::Greater),
        ("<", SkillTriggerOperator::Lesser),
    ];
    let mut found = None;
    let mut i = 0;
    let b = term.as_bytes();
    while i < b.len() {
        if let Some(&(s, op)) = OPS.iter().find(|(s, _)| term[i..].starts_with(s)) {
            if found.is_some() {
                return None;
            }
            found = Some((i, s.len(), op));
            i += s.len();
        } else {
            i += 1;
        }
    }
    let (pos, len, op) = found?;
    let v = term[pos + len..].trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'));
    let value: i32 = v.parse().ok()?;
    Some((&term[..pos], op, value))
}

/// Some triggers draw RNG at construction (left to right).
pub fn create_trigger_collection(race: &mut Race, r: SkillDetailRef, text: &str) -> SkillTriggerCollection {
    if text.is_empty() {
        return SkillTriggerCollection::Null;
    }
    let start = race.horses[r.horse].skill_manager.triggers.len();
    let mut any_empty_group = false;
    for (g, part) in text.split('@').enumerate() {
        if part.is_empty() {
            any_empty_group = true;
            continue;
        }
        for term in part.split('&') {
            let t = parse_term(term).and_then(|(kw, op, v)| create_trigger(race, r, kw, op, v));
            race.horses[r.horse].skill_manager.triggers.push(SkillTrigger { group: g as u16, ..t.unwrap_or_default() });
        }
    }
    let triggers = &race.horses[r.horse].skill_manager.triggers[start..];
    let has_lot_other = triggers.iter().any(|t| matches!(t.kind, SkillTriggerKind::RandomLotOtherActivate { .. }));
    SkillTriggerCollection::Collection { start: start as u32, len: triggers.len() as u32, any_empty_group, has_lot_other, wake: Wake::NOW }
}

/// Whether condition `text` is false all race for horse `owner` on facts fixed from the start (course, ground,
/// weather, season, time, race, her running style and mood): each OR-group has such a term that is false. Terms on
/// anything else (position, HP, RNG, the field, stats that skills can buff) count as possibly true. Builds throwaway
/// triggers, which may draw RNG: use a race that won't be run.
pub fn never_true(race: &mut Race, owner: usize, text: &str) -> bool {
    use SkillTriggerKind as K;
    let r = SkillDetailRef { horse: owner, skill: 0, detail: 0 };
    let fixed_false = |race: &mut Race, term: &str| {
        let Some(t) = parse_term(term).and_then(|(kw, op, v)| create_trigger(race, r, kw, op, v)) else { return false };
        matches!(
            t.kind,
            K::CourseCornerCount | K::Grade | K::RaceId | K::Time | K::CourseDistance | K::IsDirtGrade | K::GroundType
                | K::GroundCondition | K::CourseRotation | K::CourseDistanceType | K::Weather | K::Season | K::RaceTrackId
                | K::RunningStyle | K::Motivation
        ) && !check_value(value_lh(race, owner, &t.kind), t.op, t.value_rh)
    };
    !text.is_empty() && text.split('@').all(|g| !g.is_empty() && g.split('&').any(|term| fixed_false(race, term)))
}

fn check_value(lh: i32, op: SkillTriggerOperator, rh: i32) -> bool {
    use SkillTriggerOperator::*;
    match op {
        Equal => lh == rh,
        NotEqual => lh != rh,
        Greater => lh > rh,
        GreaterEqual => lh >= rh,
        Lesser => lh < rh,
        LesserEqual => lh <= rh,
    }
}

/// `==` is `approximately`.
fn check_value_float(lh: f32, op: SkillTriggerOperator, rh: i32) -> bool {
    use SkillTriggerOperator::*;
    let rh = rh as f32;
    match op {
        Equal => approximately(lh, rh),
        NotEqual => !approximately(lh, rh),
        Greater => rh < lh,
        GreaterEqual => rh <= lh,
        Lesser => lh < rh,
        LesserEqual => lh <= rh,
    }
}

/// `==` is `approximately`.
fn check_value_input_float(lh: f32, op: SkillTriggerOperator, rhf: f32) -> bool {
    use SkillTriggerOperator::*;
    match op {
        Equal => approximately(lh, rhf),
        NotEqual => !approximately(lh, rhf),
        Greater => lh > rhf,
        GreaterEqual => lh >= rhf,
        Lesser => lh < rhf,
        LesserEqual => lh <= rhf,
    }
}

/// Whether the trigger is met this frame; updates the trigger's own state.
/// `others_activated`: every other term of the AND-group is activated (used by random_lot_other_activate only).
fn check_activate(race: &mut Race, r: SkillDetailRef, t: &mut SkillTrigger, others_activated: bool) -> bool {
    use SkillTriggerKind as K;
    let owner = r.horse;
    let (op, rh, was_activated) = (t.op, t.value_rh, t.is_activated);
    match &mut t.kind {
        // Draws a wiz lot when the rest of the group is met and the interval is over; holds while the group stays met.
        K::RandomLotOtherActivate { is_lot_ok, lot_interval, lot_interval_max } => {
            if *is_lot_ok && others_activated {
                return true;
            }
            *is_lot_ok = false;
            if *lot_interval <= 0.0 && others_activated {
                let rnd = race.rng.random_float(100.0);
                let wiz = crate::sim::physics::stats::wiz_for(&race.horses[owner], crate::api::wiz_mechanic::SKILLS);
                *lot_interval = *lot_interval_max;
                let p = &race.data.param.Skill;
                // Evaluated in f64 as one expression, rounded at the store.
                let mut per = (p.LotActivatePerVal1 as f64 - p.LotActivatePerVal2 as f64 / wiz as f64) as f32;
                if !(p.ActivatePerMin as f32 <= per) {
                    per = p.ActivatePerMin as f32;
                }
                *is_lot_ok = rnd < per;
                return *is_lot_ok;
            }
            false
        }
        // True on the frame HP reaches 0.
        K::HpEmptyOneTime { prev_hp, is_initialized } => {
            let h = &race.horses[owner];
            if !*is_initialized {
                if !h.is_param_initialized {
                    return false;
                }
                *prev_hp = h.hp;
                *is_initialized = true;
            }
            if *prev_hp <= 0.0 {
                return false;
            }
            *prev_hp = h.hp;
            *prev_hp <= 0.0
        }
        K::ChangeOrder => was_activated || check_value(value_lh(race, owner, &t.kind), op, rh),
        // Counted on first use (the count is not reset if it bails out on an uninitialized horse).
        K::SameSkillHorseCount { same_skill_horse_count, is_initialized } => {
            if !*is_initialized {
                let id = race.horses[owner].skill_manager.skills[r.skill].skill_master.id;
                for (j, h) in race.horses.iter().enumerate() {
                    if !h.is_skill_initialized {
                        return false;
                    }
                    // A silent skill counts only for its own owner, whose race it is watching.
                    if skill::find_skill(&h.skill_manager, id).is_some_and(|k| j == owner || !h.skill_manager.is_silent(k)) {
                        *same_skill_horse_count += 1;
                    }
                }
                *is_initialized = true;
            }
            check_value(*same_skill_horse_count, op, rh)
        }
        // Latches once a detail of this skill is in the previous-frame activation list.
        K::ActivateOtherSkillDetail(enabled) => {
            if *enabled {
                return true;
            }
            let id = race.horses[owner].skill_manager.skills[r.skill].skill_master.id;
            let sm = &race.horses[owner].skill_manager;
            if sm.prev_activated_details.iter().any(|d| race.horses[d.horse].skill_manager.skills[d.skill].skill_master.id == id) {
                *enabled = true;
            }
            *enabled
        }
        // 1 when the owner activated another skill since the last check.
        K::IsActivateAnySkill { prev_activate_skill_count, activate_this_skill_count } => {
            let cur = activate_skill_count(race, owner) - *activate_this_skill_count;
            let prev = *prev_activate_skill_count;
            *prev_activate_skill_count = cur;
            if prev < cur {
                *activate_this_skill_count += 1;
            }
            check_value((prev < cur) as i32, op, rh)
        }
        // True once, on the first check in the last straight.
        K::LastStraightOneTime(prev) => {
            if !*prev {
                *prev = ai::around::is_straight_last(&race.horses[owner]);
                return *prev;
            }
            false
        }
        _ => check_pure(race, owner, t),
    }
}

/// Whether a pure trigger (see [`is_pure`]; also the stateless checks of the init-pending kinds) is met.
fn check_pure(race: &Race, owner: usize, t: &SkillTrigger) -> bool {
    use SkillTriggerKind as K;
    let (op, rh) = (t.op, t.value_rh);
    match &t.kind {
        K::Null => false,
        K::Always => true,
        K::RandomLot(ok) => *ok,
        // Truncating cast.
        K::RaceAccumulateTimeRandom(start) => *start == f32_to_i32(race.accumulate_time_since_start),
        K::OrderRateInContinue(ty) => check_value(race.horses[owner].order_in_continue[*ty] as i32, op, rh),
        K::OrderRateOutContinue(ty) => check_value(race.horses[owner].order_out_continue[*ty] as i32, op, rh),
        K::OrderLastSpurt => race.horses[owner].last_spurt_order >= 0 && check_value(value_lh(race, owner, &t.kind), op, rh),
        K::OrderFinalCornerEnd => {
            race.horses[owner].final_corner_end_order != -1 && check_value(value_lh(race, owner, &t.kind), op, rh)
        }
        K::HpPer | K::HpEmpty => race.horses[owner].is_param_initialized && check_value(value_lh(race, owner, &t.kind), op, rh),
        K::Hp => race.horses[owner].is_param_initialized && check_value_float(race.horses[owner].hp, op, rh),
        K::DistanceDiffInfront => race.horses[owner].cur_order != 0 && check_value(value_lh(race, owner, &t.kind), op, rh),
        K::DistanceDiffBehind | K::HorseBehindIn | K::HorseBehindOut => {
            race.horses[owner].horse_data.horse_index != last_horse_index(race) && check_value(value_lh(race, owner, &t.kind), op, rh)
        }
        // Gap in horse lengths (2.5 m).
        K::BashinDiffInfront => {
            let h = &race.horses[owner];
            if h.cur_order == 0 {
                return false;
            }
            let d = race.horses[horse_by_order(race, h.cur_order - 1)].distance - h.distance;
            check_value_float(if d >= 0.0 { d / 2.5 } else { 0.0 }, op, rh)
        }
        K::BashinDiffBehind => {
            let h = &race.horses[owner];
            if h.horse_data.horse_index == last_horse_index(race) {
                return false;
            }
            let d = h.distance - race.horses[horse_by_order(race, h.cur_order + 1)].distance;
            check_value_float(if d >= 0.0 { d / 2.5 } else { 0.0 }, op, rh)
        }
        // Floored to 0.1 m in f32.
        K::DistanceDiffTopFloat(rhf) => {
            let d = race.horses[first_horse(race)].distance - race.horses[owner].distance;
            check_value_input_float((d * 10.0).floor() / 10.0, op, *rhf)
        }
        K::LastSpurt => race.horses[owner].ai.last_spurt_calc_count > 0 && check_value(value_lh(race, owner, &t.kind), op, rh),
        K::ExistCharaId(e) => match op {
            SkillTriggerOperator::Equal => *e,
            SkillTriggerOperator::NotEqual => !*e,
            _ => false,
        },
        K::UsedParticularSkillId(id) => race.horses[owner].skill_manager.used_skill_ids.contains(id),
        // A non-teammate's previous-frame activation has a buff of ability type `target_type`.
        K::OtherCharacterActivateAdvantageSkill(target_type) => {
            let b = race.horses[owner].horse_data.team_id;
            for (hi, details) in &race.field.prev_activated_skills {
                if *hi == owner {
                    continue;
                }
                let a = race.horses[*hi].horse_data.team_id;
                if a != 0 && b != 0 && a == b {
                    continue;
                }
                for d in details {
                    let det = skill::detail(race, *d);
                    for ab in &det.abilities {
                        // An ability without effect reads as type 0, value 0.
                        let (ty, v) = ab.as_ref().map_or((0, 0.0), |x| (x.ability_type, x.ability_value_on_activate));
                        if ty != *target_type {
                            continue;
                        }
                        if ty == 9 {
                            if v > 0.0 {
                                return true;
                            }
                        } else if !skill::is_skill_demerit(ty, v) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        K::ActivateSkillTagGroup(tags) => race.horses[owner]
            .skill_manager
            .prev_activated_details
            .iter()
            .any(|d| skill::tag_ids(&race.horses[d.horse].skill_manager.skills[d.skill].skill_master).iter().any(|x| tags.contains(x))),
        K::FinalCornerLaterHalf { is_initialized, is_exist_final_corner, distance_min, distance_max } => {
            if !(*is_initialized && *is_exist_final_corner) {
                return false;
            }
            let d = race.horses[owner].distance;
            check_value((*distance_min <= d && d <= *distance_max) as i32, op, rh)
        }
        K::Distance(_, w) => {
            let d = race.horses[owner].distance;
            w.is_initialized && w.has_distance && w.distance_min <= d && d <= w.distance_max
        }
        K::AllCornerRandom { distance_set_list, is_initialized } => {
            if !*is_initialized || distance_set_list.is_empty() {
                return false;
            }
            let d = race.horses[owner].distance;
            for &(lo, hi) in distance_set_list.iter() {
                if d < lo {
                    return false;
                }
                if d <= hi {
                    return true;
                }
            }
            false
        }
        _ => check_value(value_lh(race, owner, &t.kind), op, rh),
    }
}

/// Reserved details always activate.
fn trigger_update(race: &mut Race, r: SkillDetailRef, t: &mut SkillTrigger, dt: f32, reserved: bool, others_activated: bool) {
    use SkillTriggerKind as K;
    if matches!(t.kind, K::Null) {
        return;
    }
    t.is_activated = check_activate(race, r, t, others_activated);
    if reserved {
        t.is_activated = true;
    }
    let value_rh = t.value_rh;
    match &mut t.kind {
        // Retry a failed window pick after the evaluation.
        K::Distance(init, w) if init.retries_in_update() && !w.is_initialized => init_distance(race, *init, value_rh, w),
        K::AllCornerRandom { distance_set_list, is_initialized } if !*is_initialized => {
            all_corner_random_init(race, distance_set_list);
            *is_initialized = true;
        }
        K::RandomLotOtherActivate { lot_interval, .. } => {
            if *lot_interval > 0.0 {
                *lot_interval -= dt;
            }
        }
        // One-time window init after the first evaluation.
        K::FinalCornerLaterHalf { is_initialized, is_exist_final_corner, distance_min, distance_max } if !*is_initialized => {
            *is_initialized = true;
            match calc_final_corner_distance(race) {
                Some((s, e)) => {
                    *is_exist_final_corner = true;
                    *distance_min = (e - s) * 0.5 + s;
                    *distance_max = e;
                }
                None => *is_exist_final_corner = false,
            }
        }
        _ => {}
    }
}

/// Whether updating the trigger only sets `is_activated`: no RNG, no own state, no read of its previous
/// `is_activated`, no one-time init still pending.
fn is_pure(kind: &SkillTriggerKind) -> bool {
    use SkillTriggerKind as K;
    match kind {
        K::RandomLotOtherActivate { .. }
        | K::HpEmptyOneTime { .. }
        | K::ChangeOrder
        | K::SameSkillHorseCount { .. }
        | K::ActivateOtherSkillDetail(_)
        | K::IsActivateAnySkill { .. }
        | K::LastStraightOneTime(_) => false,
        K::Distance(init, w) => w.is_initialized || !init.retries_in_update(),
        K::AllCornerRandom { is_initialized, .. } | K::FinalCornerLaterHalf { is_initialized, .. } => *is_initialized,
        _ => true,
    }
}

/// The wake of one pure term from its current state; `None` for an impure one. Phase and distance never decrease,
/// so a term whose window has passed never comes true again.
fn term_wake(race: &Race, owner: usize, t: &SkillTrigger) -> Option<Wake> {
    use SkillTriggerKind as K;
    use SkillTriggerOperator::*;
    if !is_pure(&t.kind) {
        return None;
    }
    let h = &race.horses[owner];
    let (op, rh) = (t.op, t.value_rh);
    // `lh op rh` for an lh that only grows, true from `from` on (None: never again).
    let growing = |lh: i32, from: Option<i32>| -> Option<i32> {
        match op {
            Equal if lh > rh => None,
            Equal | GreaterEqual => Some(rh),
            Greater => Some(rh.saturating_add(1)),
            Lesser if lh >= rh => None,
            LesserEqual if lh > rh => None,
            _ => from,
        }
    };
    let never_or = |x: Option<Wake>| x.unwrap_or(Wake::NEVER);
    Some(match &t.kind {
        // Fixed for the race: true now or never.
        K::CourseCornerCount | K::Grade | K::RaceId | K::Time | K::CourseDistance | K::IsDirtGrade | K::GateNumber
        | K::PostNumber | K::BaseSpeed | K::BaseStamina | K::BasePower | K::BaseGuts | K::BaseWiz | K::GroundType
        | K::GroundCondition | K::CourseRotation | K::CourseDistanceType | K::Weather | K::Season | K::RaceTrackId
        | K::RunningStyle | K::Motivation | K::Popularity | K::Latched(_) => {
            if check_value(value_lh(race, owner, &t.kind), op, rh) { Wake::NOW } else { Wake::NEVER }
        }
        K::RandomLot(ok) => if *ok { Wake::NOW } else { Wake::NEVER },
        K::Null => Wake::NEVER,
        K::Phase => never_or(growing(h.phase as i32, Some(i32::MIN)).map(Wake::phase)),
        // f32_to_i32(distance) >= n  <=>  distance >= n.
        K::HorseDistanceRate => never_or(growing(f32_to_i32(h.distance), Some(i32::MIN)).map(|n| Wake::distance(n as f32))),
        // course - floor(distance) op rh, with floor(distance) growing: mirror the operator.
        K::HorseRemainDistance => {
            let cd = race.info.race_course_set.distance;
            let f = crate::sim::util::floor_to_int(h.distance);
            let from = match op {
                LesserEqual => Some(cd - rh),
                Lesser => Some(cd - rh + 1),
                Equal if f > cd - rh => None,
                Equal => Some(cd - rh),
                GreaterEqual if f > cd - rh => None,
                Greater if f >= cd - rh => None,
                _ => Some(i32::MIN),
            };
            never_or(from.map(|n| if n == i32::MIN { Wake::NOW } else { Wake::distance(n as f32) }))
        }
        K::Distance(_, w) => {
            if !(w.is_initialized && w.has_distance) || h.distance > w.distance_max {
                Wake::NEVER
            } else {
                Wake::distance(w.distance_min)
            }
        }
        K::AllCornerRandom { distance_set_list, .. } => match (distance_set_list.first(), distance_set_list.last()) {
            (Some(&(lo, _)), Some(&(_, hi))) if !(h.distance > hi) => Wake::distance(lo),
            _ => Wake::NEVER,
        },
        _ => Wake::NOW,
    })
}

/// The collection's wake from its terms' current state (`None` if any term is impure).
fn collection_wake(race: &Race, owner: usize, triggers: &[SkillTrigger]) -> Option<Wake> {
    let mut wake = Wake::NEVER;
    for g in triggers.chunk_by(|a, b| a.group == b.group) {
        let mut gw = Wake::NOW;
        for t in g {
            gw = gw.and(term_wake(race, owner, t)?);
        }
        wake = wake.or(gw);
    }
    Some(wake)
}

/// Updates every trigger of every group, then returns whether some group is fully activated. Once a term of an
/// AND-group fails, its later pure terms are skipped: their stale `is_activated` is never read, as the group is
/// false anyway and no random_lot_other_activate term reads its siblings.
/// The owner's trigger arena is taken out while the triggers run (only this function reads triggers).
/// `reserved`: [`skill::detail_is_reserved_random_activate`].
pub fn collection_update(race: &mut Race, r: SkillDetailRef, slot: TriggerSlot, dt: f32, reserved: bool) -> bool {
    let d = skill::detail(race, r);
    let c = match slot {
        TriggerSlot::Pre => &d.pre_trigger,
        TriggerSlot::Main => &d.trigger,
    };
    let SkillTriggerCollection::Collection { any_empty_group, has_lot_other, wake, .. } = *c else { return false };
    // Asleep: no term would change state, and all are false.
    if !reserved && wake.is_asleep(race, r.horse) {
        return false;
    }
    let range = c.range();
    let mut arena = std::mem::take(&mut race.horses[r.horse].skill_manager.triggers);
    let triggers = &mut arena[range.clone()];
    // Groups in order, then triggers in order = the flat list in order.
    let mut start = 0; // first trigger of the current group
    let mut group_failed = false;
    let mut first_failed = None;
    for k in 0..triggers.len() {
        let g = triggers[k].group;
        if triggers[start].group != g {
            start = k;
            group_failed = false;
        }
        if group_failed && !has_lot_other && is_pure(&triggers[k].kind) {
            continue;
        }
        // The other group members' current `is_activated`.
        let others = matches!(triggers[k].kind, SkillTriggerKind::RandomLotOtherActivate { .. })
            && triggers[start..].iter().take_while(|t| t.group == g).enumerate().all(|(j, t)| start + j == k || t.is_activated);
        trigger_update(race, r, &mut triggers[k], dt, reserved, others);
        group_failed |= !triggers[k].is_activated;
        if group_failed && first_failed.is_none() {
            first_failed = Some(k);
        }
    }
    let any = collection_is_activated_any(triggers, any_empty_group);
    let new_wake = match collection_wake(race, r.horse, triggers) {
        Some(w) if !(any || any_empty_group || has_lot_other) => {
            let one_group = triggers.first().map(|t| t.group) == triggers.last().map(|t| t.group);
            match first_failed {
                // A false term of the only group: the collection stays false while it does.
                Some(k) if one_group => Wake { sentinel: (range.start + k) as u32, ..w },
                _ => w,
            }
        }
        _ => Wake::NOW,
    };
    race.horses[r.horse].skill_manager.triggers = arena;
    let d = skill::detail_mut(race, r);
    if let SkillTriggerCollection::Collection { wake, .. } = match slot {
        TriggerSlot::Pre => &mut d.pre_trigger,
        TriggerSlot::Main => &mut d.trigger,
    } {
        *wake = new_wake;
    }
    any
}

fn collection_is_activated_any(triggers: &[SkillTrigger], any_empty_group: bool) -> bool {
    any_empty_group || triggers.chunk_by(|a, b| a.group == b.group).any(|g| g.iter().all(|t| t.is_activated))
}

/// A detail with an is_activate_other_skill_detail term runs its own cooldown, and its activation does not count as
/// a skill activation.
pub fn collection_has_activate_other_skill_detail(arena: &[SkillTrigger], c: &SkillTriggerCollection) -> bool {
    arena[c.range()].iter().any(|t| matches!(t.kind, SkillTriggerKind::ActivateOtherSkillDetail(_)))
}

/// `v` % of `n`, rounded half up (f32, truncating cast).
pub(super) fn rescale_rate(v: i32, n: i32) -> i32 {
    f32_to_i32(v as f32 / 100.0 * n as f32 + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::skill::trigger::kind::order_type;
    use crate::data::{DataEra, GameData};
    use std::path::Path;

    #[test]
    fn never_true_fixed_terms() {
        use crate::sim::defs::RunningStyle;
        let mut race = crate::sim::test_util::ai_race(2);
        race.info.horse_index_by_popularity = vec![0, 1];
        race.horses[0].horse_data.running_style = RunningStyle::Sashi; // 3
        assert!(never_true(&mut race, 0, "running_style==1&phase>=2"));
        assert!(never_true(&mut race, 0, "running_style==1@running_style==4"));
        assert!(!never_true(&mut race, 0, "running_style==3&phase>=2"));
        assert!(!never_true(&mut race, 0, "running_style==1@phase>=2")); // the other group may come true
        assert!(!never_true(&mut race, 0, "running_style==1@")); // an empty group is true
        assert!(!never_true(&mut race, 0, ""));
    }

    #[test]
    fn parse_terms() {
        assert_eq!(parse_term("distance_rate>=50"), Some(("distance_rate", SkillTriggerOperator::GreaterEqual, 50)));
        assert_eq!(parse_term("change_order_onetime<0"), Some(("change_order_onetime", SkillTriggerOperator::Lesser, 0)));
        assert_eq!(parse_term("order!=-1"), Some(("order", SkillTriggerOperator::NotEqual, -1)));
        assert_eq!(parse_term("order==1 "), Some(("order", SkillTriggerOperator::Equal, 1)));
        assert_eq!(parse_term("order"), None);
        assert_eq!(parse_term("order==x"), None);
        assert_eq!(parse_term("a==1==2"), None);
        assert_eq!(order_type("40"), Some(3));
        assert_eq!(rescale_rate(50, 9), 5); // 4.5 + 0.5 → 5
        assert_eq!(rescale_rate(40, 9), 4); // 3.6 + 0.5 = 4.1 → 4
    }

    /// Every condition term of the current master (and of every era snapshot) must parse and build a trigger.
    #[test]
    fn all_master_conditions_parse() {
        let dir = Path::new(crate::data::DATA_DIR);
        if !dir.exists() {
            return;
        }
        let mut race = crate::sim::test_util::ai_race(2);
        race.horses[0].skill_manager.skills.push(Default::default());
        race.info.horse_index_by_popularity = vec![0, 1];
        let r = SkillDetailRef { horse: 0, skill: 0, detail: 0 };
        let (mut bad, mut checked) = (Vec::new(), 0);
        let mut eras = vec![DataEra::current()];
        for d in ["20260210", "20260620"] {
            eras.push(DataEra::for_date(d).unwrap());
        }
        for era in eras {
            let g = GameData::load(dir, era).unwrap();
            for (id, s) in &g.skills {
                for d in &s.details {
                    for text in [&d.precondition, &d.condition] {
                        for term in text.split(['@', '&']).filter(|t| !t.is_empty()) {
                            checked += 1;
                            let built = parse_term(term).and_then(|(kw, op, rh)| create_trigger(&mut race, r, kw, op, rh));
                            if built.is_none() {
                                bad.push(format!("{id}: {term:?} in {text:?}"));
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 1000, "only {checked} terms checked");
        assert!(bad.is_empty(), "unknown/unparsable condition terms:\n{}", bad.join("\n"));
    }
}
