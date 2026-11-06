//! A race's skills for the GPU: the skill program (static for the race: parsed conditions, abilities, lookup tables
//! and where each piece of state lives) and the skill state (timers, latches, lists, modifier handles). Formats in
//! `skill_spec.rs`.
//!
//! What the CPU keeps only for speed is left out: a condition's wake thresholds and the per-horse skill walk. A term
//! whose `is_activated` nothing reads later (a pure term outside a group with `random_lot_other_activate`; only
//! `change_order` reads its own) is written as 0, because the CPU skips updating pure terms of a failed group. The
//! targets an ability picked are not kept either: only the activation itself uses them.

include!(concat!(env!("OUT_DIR"), "/skill_gen.rs"));

use crate::sim::race::Race;
use crate::sim::skill::ability::SkillAbilityAdditionalActivate;
use crate::sim::skill::trigger::kind::SkillTriggerKind as K;
use crate::sim::skill::trigger::window::{DistanceInit, calc_final_corner_distance};
use crate::sim::skill::trigger::{SkillTrigger, SkillTriggerCollection};
use crate::sim::skill::{SkillDetailRef, tag_ids, value};

pub use crate::sim::skill::ability::ModifierHandle;

/// A race's skill program and skill state, and a name for each state word (empty from
/// [`pack_skills_unlabeled`]).
pub struct SkillPack {
    pub program: Vec<u32>,
    pub state: Vec<u32>,
    pub labels: Vec<String>,
}

struct B {
    p: Vec<u32>,
    s: Vec<u32>,
    labels: Option<Vec<String>>,
}

impl B {
    fn prog(&mut self, n: u32) -> u32 {
        let at = self.p.len() as u32;
        self.p.resize(self.p.len() + n as usize, 0);
        at
    }
    /// A label, formatted only when labels are kept.
    fn lab(&self, a: std::fmt::Arguments) -> String {
        if self.labels.is_some() { a.to_string() } else { String::new() }
    }

    fn state(&mut self, label: &str, fields: &[&str]) -> u32 {
        let at = self.s.len() as u32;
        for f in fields {
            self.s.push(0);
            if let Some(l) = &mut self.labels {
                l.push(format!("{label} {f}"));
            }
        }
        at
    }
    fn list(&mut self, label: &str, cap: usize) -> u32 {
        let at = self.s.len() as u32;
        self.s.push(0);
        if let Some(l) = &mut self.labels {
            l.push(format!("{label} count"));
            l.extend((0..cap).map(|k| format!("{label}[{k}]")));
        }
        self.s.resize(self.s.len() + cap, 0);
        at
    }
    fn set_list(&mut self, at: u32, v: impl IntoIterator<Item = u32>) {
        let mut n = 0;
        for x in v {
            self.s[(at + 1 + n) as usize] = x;
            n += 1;
        }
        self.s[at as usize] = n;
    }
    fn table(&mut self, pairs: &[(f32, f32)]) -> (u32, u32) {
        let at = self.prog(2 * pairs.len() as u32);
        for (k, &(t, v)) in pairs.iter().enumerate() {
            self.p[at as usize + 2 * k] = t.to_bits();
            self.p[at as usize + 2 * k + 1] = v.to_bits();
        }
        (at, pairs.len() as u32)
    }
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// A detail as the state's lists hold it: skill * 2 + detail.
fn detail_code(r: &SkillDetailRef) -> u32 {
    (r.skill * 2 + r.detail) as u32
}

/// The race master row's (grade, id, is_dirtgrade).
fn race_master(race: &Race) -> (i32, i32, i32) {
    let m = race.data.master.race_instance.get(&race.input.race_instance_id).and_then(|i| race.data.master.race.get(&i.race_id));
    m.map_or((0, 0, 0), |m| (m.grade, m.id, m.is_dirtgrade))
}

fn asc<T>(arr: &[T], key: impl Fn(&T) -> (f32, f32)) -> Vec<(f32, f32)> {
    arr.iter().map(key).collect()
}

/// Kind code, static payload (p0, p1, p2) and state words after `is_activated`.
fn kind_words(race: &Race, b: &mut B, kind: &K) -> (u32, [u32; 3], Vec<u32>) {
    let n = [0u32; 3];
    let b32 = |x: bool| x as u32;
    let (code, p, state): (u32, [u32; 3], Vec<u32>) = match kind {
        K::Null => (K_NULL, n, vec![]),
        K::CourseCornerCount => (K_COURSE_CORNER_COUNT, n, vec![]),
        K::Grade => (K_GRADE, n, vec![]),
        K::RaceId => (K_RACE_ID, n, vec![]),
        K::Time => (K_TIME, n, vec![]),
        K::CourseDistance => (K_COURSE_DISTANCE, n, vec![]),
        K::IsDirtGrade => (K_IS_DIRT_GRADE, n, vec![]),
        K::RaceAccumulateTime => (K_RACE_ACCUMULATE_TIME, n, vec![]),
        K::Slope => (K_SLOPE, n, vec![]),
        K::Corner => (K_CORNER, n, vec![]),
        K::Furlong => (K_FURLONG, n, vec![]),
        K::Order => (K_ORDER, n, vec![]),
        K::OrderLastSpurt => (K_ORDER_LAST_SPURT, n, vec![]),
        K::OrderFinalCornerEnd => (K_ORDER_FINAL_CORNER_END, n, vec![]),
        K::GateNumber => (K_GATE_NUMBER, n, vec![]),
        K::PostNumber => (K_POST_NUMBER, n, vec![]),
        K::HorseDistanceRate => (K_HORSE_DISTANCE_RATE, n, vec![]),
        K::HorseRemainDistance => (K_HORSE_REMAIN_DISTANCE, n, vec![]),
        K::HorseDistanceDiffRate => (K_HORSE_DISTANCE_DIFF_RATE, n, vec![]),
        K::OtherHorseBehind => (K_OTHER_HORSE_BEHIND, n, vec![]),
        K::BeBlocked => (K_BE_BLOCKED, n, vec![]),
        K::BeBlockedFront => (K_BE_BLOCKED_FRONT, n, vec![]),
        K::BeBlockedSide => (K_BE_BLOCKED_SIDE, n, vec![]),
        K::BeBlockedAllContinueTime => (K_BE_BLOCKED_ALL_CONTINUE_TIME, n, vec![]),
        K::BeBlockedFrontContinueTime => (K_BE_BLOCKED_FRONT_CONTINUE_TIME, n, vec![]),
        K::BeBlockedSideContinueTime => (K_BE_BLOCKED_SIDE_CONTINUE_TIME, n, vec![]),
        K::BeBlockedFrontHpEmpty => (K_BE_BLOCKED_FRONT_HP_EMPTY, n, vec![]),
        K::BeBlockedFrontHpEmptyContinueTime => (K_BE_BLOCKED_FRONT_HP_EMPTY_CONTINUE_TIME, n, vec![]),
        K::TemptationCount => (K_TEMPTATION_COUNT, n, vec![]),
        K::CompeteFightCount => (K_COMPETE_FIGHT_COUNT, n, vec![]),
        K::GoodStart => (K_GOOD_START, n, vec![]),
        K::BadStart => (K_BAD_START, n, vec![]),
        K::BaseSpeed => (K_BASE_SPEED, n, vec![]),
        K::BaseStamina => (K_BASE_STAMINA, n, vec![]),
        K::BasePower => (K_BASE_POWER, n, vec![]),
        K::BaseGuts => (K_BASE_GUTS, n, vec![]),
        K::BaseWiz => (K_BASE_WIZ, n, vec![]),
        K::HpPer => (K_HP_PER, n, vec![]),
        K::HpEmpty => (K_HP_EMPTY, n, vec![]),
        K::CurrentSpeed => (K_CURRENT_SPEED, n, vec![]),
        K::SpeedOrder => (K_SPEED_ORDER, n, vec![]),
        K::VisibleHorse => (K_VISIBLE_HORSE, n, vec![]),
        K::ChangeOrder => (K_CHANGE_ORDER, n, vec![]),
        K::ChangeOrderOneTime => (K_CHANGE_ORDER_ONE_TIME, n, vec![]),
        K::LaneType => (K_LANE_TYPE, n, vec![]),
        K::LanePer => (K_LANE_PER, n, vec![]),
        K::MoveLane => (K_MOVE_LANE, n, vec![]),
        K::Overtake => (K_OVERTAKE, n, vec![]),
        K::OvertakeTargetContinueTime => (K_OVERTAKE_TARGET_CONTINUE_TIME, n, vec![]),
        K::OvertakeTargetHaveNoOrderUpContinueTime => (K_OVERTAKE_TARGET_NO_ORDER_UP_TIME, n, vec![]),
        K::OvertakeTargetHaveNoOrderDownContinueTime => (K_OVERTAKE_TARGET_NO_ORDER_DOWN_TIME, n, vec![]),
        K::Surrounded => (K_SURROUNDED, n, vec![]),
        K::NearHorseCount => (K_NEAR_HORSE_COUNT, n, vec![]),
        K::NearAroundAccumulateTime => (K_NEAR_AROUND_ACCUMULATE_TIME, n, vec![]),
        K::DistanceDiffTop => (K_DISTANCE_DIFF_TOP, n, vec![]),
        K::DistanceDiffInfront => (K_DISTANCE_DIFF_INFRONT, n, vec![]),
        K::DistanceDiffBehind => (K_DISTANCE_DIFF_BEHIND, n, vec![]),
        K::NearFrontHorseCount => (K_NEAR_FRONT_HORSE_COUNT, n, vec![]),
        K::NearBehindHorseCount => (K_NEAR_BEHIND_HORSE_COUNT, n, vec![]),
        K::HorseBehindIn => (K_HORSE_BEHIND_IN, n, vec![]),
        K::HorseBehindOut => (K_HORSE_BEHIND_OUT, n, vec![]),
        K::HorseInfrontNearTime => (K_HORSE_INFRONT_NEAR_TIME, n, vec![]),
        K::HorseBehindNearTime => (K_HORSE_BEHIND_NEAR_TIME, n, vec![]),
        K::HorseInfrontNearLaneTime => (K_HORSE_INFRONT_NEAR_LANE_TIME, n, vec![]),
        K::HorseBehindNearLaneTime => (K_HORSE_BEHIND_NEAR_LANE_TIME, n, vec![]),
        K::ActivateHealSkillCount => (K_ACTIVATE_HEAL_SKILL_COUNT, n, vec![]),
        K::ActivateCount => (K_ACTIVATE_COUNT, n, vec![]),
        K::ActivateCountLaterHalf => (K_ACTIVATE_COUNT_LATER_HALF, n, vec![]),
        K::ActivateCountAllTeam => (K_ACTIVATE_COUNT_ALL_TEAM, n, vec![]),
        K::ChangeOrderUpPhaseMiddle => (K_CHANGE_ORDER_UP_PHASE_MIDDLE, n, vec![]),
        K::ChangeOrderUpPhaseEndAfter => (K_CHANGE_ORDER_UP_PHASE_END_AFTER, n, vec![]),
        K::ChangeOrderUpCorner => (K_CHANGE_ORDER_UP_CORNER, n, vec![]),
        K::ChangeOrderUpLastSpurt => (K_CHANGE_ORDER_UP_LAST_SPURT, n, vec![]),
        K::ChangeOrderUpFinalCornerAfter => (K_CHANGE_ORDER_UP_FINAL_CORNER_AFTER, n, vec![]),
        K::ChangeOrderDownPhaseMiddle => (K_CHANGE_ORDER_DOWN_PHASE_MIDDLE, n, vec![]),
        K::ChangeOrderDownPhaseEndAfter => (K_CHANGE_ORDER_DOWN_PHASE_END_AFTER, n, vec![]),
        K::ChangeOrderDownCorner => (K_CHANGE_ORDER_DOWN_CORNER, n, vec![]),
        K::ChangeOrderDownLastSpurt => (K_CHANGE_ORDER_DOWN_LAST_SPURT, n, vec![]),
        K::ChangeOrderDownFinalCornerAfter => (K_CHANGE_ORDER_DOWN_FINAL_CORNER_AFTER, n, vec![]),
        K::GroundType => (K_GROUND_TYPE, n, vec![]),
        K::GroundCondition => (K_GROUND_CONDITION, n, vec![]),
        K::CourseRotation => (K_COURSE_ROTATION, n, vec![]),
        K::CourseDistanceType => (K_COURSE_DISTANCE_TYPE, n, vec![]),
        K::Weather => (K_WEATHER, n, vec![]),
        K::Season => (K_SEASON, n, vec![]),
        K::RaceTrackId => (K_RACE_TRACK_ID, n, vec![]),
        K::StraightFrontType => (K_STRAIGHT_FRONT_TYPE, n, vec![]),
        K::FinalCorner => (K_FINAL_CORNER, n, vec![]),
        K::RunningStyle => (K_RUNNING_STYLE, n, vec![]),
        K::Motivation => (K_MOTIVATION, n, vec![]),
        K::Popularity => (K_POPULARITY, n, vec![]),
        K::Phase => (K_PHASE, n, vec![]),
        K::Temptation => (K_TEMPTATION, n, vec![]),
        K::IsLastSpurt => (K_IS_LAST_SPURT, n, vec![]),
        K::LastSpurt => (K_LAST_SPURT, n, vec![]),
        K::Always => (K_ALWAYS, n, vec![]),
        K::Latched(v) => (K_LATCHED, [*v as u32, 0, 0], vec![]),
        K::ExistCharaId(e) => (K_EXIST_CHARA_ID, [b32(*e), 0, 0], vec![]),
        K::Hp => (K_HP, n, vec![]),
        K::BashinDiffInfront => (K_BASHIN_DIFF_INFRONT, n, vec![]),
        K::BashinDiffBehind => (K_BASHIN_DIFF_BEHIND, n, vec![]),
        K::DistanceDiffTopFloat(x) => (K_DISTANCE_DIFF_TOP_FLOAT, [f(*x), 0, 0], vec![]),
        K::RandomLot(ok) => (K_RANDOM_LOT, n, vec![b32(*ok)]),
        K::RandomLotOtherActivate { is_lot_ok, lot_interval, lot_interval_max } => {
            (K_RANDOM_LOT_OTHER_ACTIVATE, [f(*lot_interval_max), 0, 0], vec![b32(*is_lot_ok), f(*lot_interval)])
        }
        K::RaceAccumulateTimeRandom(s) => (K_RACE_ACCUMULATE_TIME_RANDOM, n, vec![*s as u32]),
        K::OrderRateInContinue(t) => (K_ORDER_RATE_IN_CONTINUE, [*t as u32, 0, 0], vec![]),
        K::OrderRateOutContinue(t) => (K_ORDER_RATE_OUT_CONTINUE, [*t as u32, 0, 0], vec![]),
        K::RunningStyleTemptationCount { style, is_check_only_opponent } => {
            (K_RUNNING_STYLE_TEMPTATION_COUNT, [*style as u32, b32(*is_check_only_opponent), 0], vec![])
        }
        K::TemptationBehind { is_check_only_opponent } => (K_TEMPTATION_BEHIND, [b32(*is_check_only_opponent), 0, 0], vec![]),
        K::TemptationInfront { is_check_only_opponent } => (K_TEMPTATION_INFRONT, [b32(*is_check_only_opponent), 0, 0], vec![]),
        K::HpEmptyOneTime { prev_hp, is_initialized } => (K_HP_EMPTY_ONE_TIME, n, vec![f(*prev_hp), b32(*is_initialized)]),
        K::HorseViewerIdRemainDistance(hs) => {
            (K_HORSE_VIEWER_ID_REMAIN_DISTANCE, [hs.iter().fold(0u32, |m, &h| m | 1 << h), 0, 0], vec![])
        }
        K::HorseBehindNearLaneTimeParamSet(p) => (K_HORSE_BEHIND_NEAR_LANE_TIME_PARAM_SET, [*p as u32, 0, 0], vec![]),
        K::SameSkillHorseCount { same_skill_horse_count, is_initialized } => {
            (K_SAME_SKILL_HORSE_COUNT, n, vec![*same_skill_horse_count as u32, b32(*is_initialized)])
        }
        K::UsedParticularSkillId(id) => (K_USED_PARTICULAR_SKILL_ID, [*id as u32, 0, 0], vec![]),
        K::OtherCharacterActivateAdvantageSkill(t) => (K_OTHER_CHARACTER_ACTIVATE_ADVANTAGE_SKILL, [*t as u32, 0, 0], vec![]),
        K::ActivateOtherSkillDetail(e) => (K_ACTIVATE_OTHER_SKILL_DETAIL, n, vec![b32(*e)]),
        K::ActivateCountPhase(ps) => (K_ACTIVATE_COUNT_PHASE, [ps.iter().fold(0u32, |m, &p| m | 1 << p), 0, 0], vec![]),
        K::IsActivateAnySkill { prev_activate_skill_count, activate_this_skill_count } => {
            (K_IS_ACTIVATE_ANY_SKILL, n, vec![*prev_activate_skill_count as u32, *activate_this_skill_count as u32])
        }
        K::ActivateSkillTagGroup(tags) => {
            let at = b.prog(tags.len() as u32);
            for (k, &t) in tags.iter().enumerate() {
                b.p[at as usize + k] = t as u32;
            }
            (K_ACTIVATE_SKILL_TAG_GROUP, [at, tags.len() as u32, 0], vec![])
        }
        K::LastStraightOneTime(prev) => (K_LAST_STRAIGHT_ONE_TIME, n, vec![b32(*prev)]),
        K::FinalCornerLaterHalf { is_initialized, is_exist_final_corner, distance_min, distance_max } => {
            // What the first update computes (the course is fixed).
            let p = match calc_final_corner_distance(race) {
                Some((s, e)) => [1, f((e - s) * 0.5 + s), f(e)],
                None => [0, 0, 0],
            };
            (K_FINAL_CORNER_LATER_HALF, p, vec![b32(*is_initialized), b32(*is_exist_final_corner), f(*distance_min), f(*distance_max)])
        }
        K::Distance(init, w) => {
            use DistanceInit::*;
            let retries = matches!(init, StraightRandom | WhileInStraightRandom | LastStraightRandom | CornerRandom | FinalCornerRandom);
            (K_DISTANCE, [b32(retries), 0, 0], vec![f(w.distance_min), f(w.distance_max), b32(w.is_initialized), b32(w.has_distance)])
        }
        K::AllCornerRandom { distance_set_list, is_initialized } => {
            assert!(distance_set_list.len() <= 4);
            let mut s = vec![distance_set_list.len() as u32, b32(*is_initialized)];
            for k in 0..4 {
                let (lo, hi) = distance_set_list.get(k).copied().unwrap_or((0.0, 0.0));
                s.push(f(lo));
                s.push(f(hi));
            }
            (K_ALL_CORNER_RANDOM, n, s)
        }
    };
    (code, p, state)
}

const KIND_STATE_NAMES: &[&str] = &["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"];

/// The terms of a collection: program records and state. Returns (first term, count, flags).
fn pack_collection(race: &Race, b: &mut B, arena: &[SkillTrigger], c: &SkillTriggerCollection, label: &str) -> (u32, u32, u32) {
    let SkillTriggerCollection::Collection { start, len, any_empty_group, has_lot_other, .. } = *c else { return (0, 0, 0) };
    let first = b.prog(len * PT_WORDS);
    for k in 0..len {
        let t = &arena[(start + k) as usize];
        let (code, p, kind_state) = kind_words(race, b, &t.kind);
        let mut names = vec!["is_activated"];
        names.extend(&KIND_STATE_NAMES[..kind_state.len()]);
        let st = b.state(&b.lab(format_args!("{label} t{k}")), &names);
        let keeps_activated = has_lot_other || matches!(t.kind, K::ChangeOrder);
        b.s[st as usize] = (keeps_activated && t.is_activated) as u32;
        for (j, &w) in kind_state.iter().enumerate() {
            b.s[st as usize + 1 + j] = w;
        }
        let r = (first + k * PT_WORDS) as usize;
        b.p[r + PT_KIND as usize] = code;
        b.p[r + PT_OP as usize] = t.op as u32;
        b.p[r + PT_RH as usize] = t.value_rh as u32;
        b.p[r + PT_GROUP as usize] = t.group as u32;
        b.p[r + PT_STATE as usize] = st;
        b.p[r + PT_P0 as usize] = p[0];
        b.p[r + PT_P1 as usize] = p[1];
        b.p[r + PT_P2 as usize] = p[2];
    }
    (first, len, 1 | (any_empty_group as u32) << 1 | (has_lot_other as u32) << 2)
}

/// Builds the program and the state of the race as it stands.
pub fn pack_skills(race: &Race) -> SkillPack {
    build(race, true)
}

/// [`pack_skills`] without the labels (faster).
pub fn pack_skills_unlabeled(race: &Race) -> SkillPack {
    build(race, false)
}

fn build(race: &Race, labels: bool) -> SkillPack {
    let mut b = B { p: Vec::new(), s: Vec::new(), labels: labels.then(Vec::new) };
    let n = race.horses.len();
    let sp = &race.data.param.Skill;
    b.prog(PR_WORDS);
    let (grade, race_id, dirt) = race_master(race);
    let cs = &race.info.race_course_set;
    let unsupported = race.input.horses.iter().any(|h| !h.observed_skills.is_empty() || !h.muted_skills.is_empty());
    for (k, v) in [
        (PR_HORSE_COUNT, n as u32),
        (PR_FLAGS, unsupported as u32),
        (PR_CORNER_COUNT, race.course.corners.len() as u32),
        (PR_GRADE, grade as u32),
        (PR_RACE_ID, race_id as u32),
        (PR_TIME, race.info.time as u32),
        (PR_IS_DIRT_GRADE, dirt as u32),
        (PR_GROUND, cs.ground as u32),
        (PR_GROUND_CONDITION, race.info.ground_condition as u32),
        (PR_TURN, cs.turn as u32),
        (PR_WEATHER, race.info.weather as u32),
        (PR_SEASON, race.info.season as u32),
        (PR_TRACK_ID, cs.race_track_id as u32),
        (PR_FIELD_HORSE_NUM, race.field.horse_num as u32),
    ] {
        b.p[k as usize] = v;
    }
    let (at, c) = b.table(&asc(&sp.AbiiltyValueUsageRandom1Array, |e| (e.AbilityValueCoef, e.Per as f32)));
    b.p[PR_RANDOM1 as usize] = at;
    b.p[PR_RANDOM1_COUNT as usize] = c;
    let (at, c) = b.table(&asc(&sp.AbiiltyValueUsageRandom2Array, |e| (e.AbilityValueCoef, e.Per as f32)));
    b.p[PR_RANDOM2 as usize] = at;
    b.p[PR_RANDOM2_COUNT as usize] = c;
    let heal = b.list("field heal_skill_horses", crate::gpu::layout::MAX_HORSES as usize);
    b.set_list(heal, race.field.heal_skill_horses.iter().map(|&h| h as u32));
    b.p[PR_HEAL_LIST as usize] = heal;
    let horses = b.prog(n as u32 * PH_WORDS);
    b.p[PR_HORSES as usize] = horses;
    let tag_group1 = &sp.AbilityValueUsageTagGroup1Array;
    for i in 0..n {
        let h = &race.horses[i];
        let sm = &h.skill_manager;
        assert!(sm.observed.is_none() || unsupported);
        let ph = (horses + i as u32 * PH_WORDS) as usize;
        let ns = sm.skills.len();
        let hl = b.lab(format_args!("h{i}"));
        let used = b.list(&b.lab(format_args!("{hl} used_skill_ids")), ns);
        b.set_list(used, sm.used_skill_ids.iter().map(|&x| x as u32));
        let prev = b.list(&b.lab(format_args!("{hl} prev_activated_details")), 2 * ns);
        b.set_list(prev, sm.prev_activated_details.iter().map(|r| detail_code(r)));
        let act = b.list(&b.lab(format_args!("{hl} activate_skills")), 2 * ns);
        b.set_list(act, sm.activate_skills.iter().map(|r| detail_code(r)));
        let res = b.list(&b.lab(format_args!("{hl} reserved_skill_ids")), ns);
        b.set_list(res, sm.reserved_skill_ids.iter().map(|&x| x as u32));
        let cached = b.list(&b.lab(format_args!("{hl} field prev_activated_skills")), 2 * ns);
        let cached_list = race.field.prev_activated_skills.iter().find(|(j, _)| *j == i).map_or(&[][..], |(_, d)| &d[..]);
        for r in cached_list {
            assert_eq!(r.horse, i);
        }
        b.set_list(cached, cached_list.iter().map(detail_code));
        for r in sm.prev_activated_details.iter().chain(&sm.activate_skills) {
            assert_eq!(r.horse, i);
        }
        let hd = &h.horse_data;
        let rp = &hd.race_param;
        let team = race.field.teams.iter().find(|t| t.team_id == hd.team_id && hd.team_id != 0).map_or(&[][..], |t| &t.members[..]);
        let team_at = b.prog(team.len() as u32);
        for (k, &m) in team.iter().enumerate() {
            b.p[team_at as usize + k] = m as u32;
        }
        let skills = b.prog(ns as u32 * PS_WORDS);
        for (k, v) in [
            (PH_SKILL_COUNT, ns as u32),
            (PH_SKILLS, skills),
            (PH_USED_IDS, used),
            (PH_PREV_DETAILS, prev),
            (PH_ACTIVATE_DETAILS, act),
            (PH_RESERVED_IDS, res),
            (PH_CACHED_DETAILS, cached),
            (PH_TEAM_ID, hd.team_id as u32),
            (PH_CHARA_ID, hd.chara_id as u32),
            (PH_POST_NUMBER, hd.post_number as u32),
            (PH_POPULARITY, hd.popularity as u32),
            (PH_FINAL_GRADE, hd.final_grade as u32),
            (PH_BASE_SPEED, f(rp.base_speed)),
            (PH_BASE_STAMINA, f(rp.base_stamina)),
            (PH_BASE_POW, f(rp.base_pow)),
            (PH_BASE_GUTS, f(rp.base_guts)),
            (PH_BASE_WIZ, f(rp.base_wiz)),
            (PH_BASE_WIZ_SKILLS, f(crate::sim::physics::stats::base_wiz_for(h, crate::api::wiz_mechanic::SKILLS))),
            (PH_MOTIVATION, rp.motivation as u32),
            (PH_TEAM, team_at),
            (PH_TEAM_COUNT, team.len() as u32),
        ] {
            b.p[ph + k as usize] = v;
        }
        for (s, base) in sm.skills.iter().enumerate() {
            let ps = (skills + s as u32 * PS_WORDS) as usize;
            let m = &base.skill_master;
            let sl = b.lab(format_args!("{hl} s{s} ({})", base.id));
            let st = b.state(&sl, &["cool_down_time", "is_activate_enable"]);
            b.s[st as usize] = f(base.cool_down_time);
            b.s[st as usize + 1] = base.is_activate_enable as u32;
            let tags = tag_ids(m);
            let tags_at = b.prog(tags.len() as u32);
            for (k, &t) in tags.iter().enumerate() {
                b.p[tags_at as usize + k] = t as u32;
            }
            let specific = !tag_group1.is_empty() && tags.iter().any(|t| tag_group1.contains(t));
            let mut det_at = [0u32; 2];
            for (d, det) in base.details.iter().enumerate() {
                let pd = b.prog(PD_WORDS);
                det_at[d] = pd;
                let pd = pd as usize;
                let dl = b.lab(format_args!("{sl} d{d}"));
                let ad = b.state(&dl, &["is_activated", "left_time", "cool_down_time", "need_pre", "increment_count", "additional_time"]);
                for (k, v) in [
                    (AD_IS_ACTIVATED, det.is_activated as u32),
                    (AD_LEFT_TIME, f(det.left_time)),
                    (AD_COOL_DOWN_TIME, f(det.cool_down_time)),
                    (AD_NEED_PRE, det.is_need_pre_trigger_check as u32),
                    (AD_INCREMENT_COUNT, det.time.increment_count as u32),
                    (AD_ADDITIONAL_TIME, f(det.time.additional_ability_time)),
                ] {
                    b.s[(ad + k) as usize] = v;
                }
                let (pre_l, cond_l) = (b.lab(format_args!("{dl} pre")), b.lab(format_args!("{dl} cond")));
                let (pre, pre_n, pre_f) = pack_collection(race, &mut b, &sm.triggers, &det.pre_trigger, &pre_l);
                let (main, main_n, main_f) = pack_collection(race, &mut b, &sm.triggers, &det.trigger, &cond_l);
                let t = &det.time;
                let time_table: Vec<(f32, f32)> = match t.usage {
                    2 => vec![
                        (sp.AbilityTimeDiffDistanceTopCoefPerDistance, sp.AbilityTimeDiffDistanceTopCoefMin),
                        (sp.AbilityTimeDiffDistanceTopCoefMax, 0.0),
                    ],
                    3 => asc(&sp.TimeRemainHpCoefAscArray, |e| (e.HpThreshold, e.Multiply)),
                    7 => asc(&sp.TimeRemainHpCoefAscArray2, |e| (e.HpThreshold, e.Multiply)),
                    5 => asc(&sp.BlockedSideMaxContinueTimePhaseMiddleRunAbilityTimeCoefAscArray1, |e| {
                        (e.BlockedSideMaxContinueTimeThreshold as f32, e.Multiply)
                    }),
                    6 => asc(&sp.BlockedSideMaxContinueTimePhaseMiddleRunAbilityTimeCoefAscArray2, |e| {
                        (e.BlockedSideMaxContinueTimeThreshold as f32, e.Multiply)
                    }),
                    4 => vec![(sp.OrderUpAddAbilityTimeMaxCount as f32, sp.OrderUpAddAbilityTime * t.ability_time_by_distance_rate)],
                    _ => vec![],
                };
                let (tt, ttn) = b.table(&time_table);
                let abilities = b.prog(det.abilities.len() as u32 * PA_WORDS);
                for (a, slot) in det.abilities.iter().enumerate() {
                    let pa = (abilities + a as u32 * PA_WORDS) as usize;
                    let Some(ab) = slot else { continue };
                    let al = b.lab(format_args!("{dl} a{a}"));
                    let aa = b.state(&al, &["value_on_activate", "additional_activate_count", "aa_count", "run_count"]);
                    let run_names: Vec<String> = if b.labels.is_some() {
                        (0..MAX_RUNS).flat_map(|k| [format!("run{k} first"), format!("run{k} last")]).collect()
                    } else {
                        vec![String::new(); 2 * MAX_RUNS as usize]
                    };
                    let runs = b.state(&al, &run_names.iter().map(String::as_str).collect::<Vec<_>>());
                    assert_eq!(runs, aa + AA_RUNS);
                    let (aa_kind, aa_max, aa_count) = match ab.ability_additional_activate {
                        SkillAbilityAdditionalActivate::Null => (0, 0, 0),
                        SkillAbilityAdditionalActivate::OrderUp { additional_activate_max_count, additional_activate_count } => {
                            (1, additional_activate_max_count, additional_activate_count)
                        }
                        SkillAbilityAdditionalActivate::ActivateAnySkill { additional_activate_max_count, additional_activate_count } => {
                            (2, additional_activate_max_count, additional_activate_count)
                        }
                    };
                    let id_runs = modifier_runs(&ab.modifiers);
                    assert!(id_runs.len() <= MAX_RUNS as usize, "more modifier id runs than the GPU keeps");
                    for (k, v) in [
                        (AA_VALUE_ON_ACTIVATE, f(ab.ability_value_on_activate)),
                        (AA_ADDITIONAL_ACTIVATE_COUNT, ab.additional_activate_count as u32),
                        (AA_AA_COUNT, aa_count as u32),
                        (AA_RUN_COUNT, id_runs.len() as u32),
                    ] {
                        b.s[(aa + k) as usize] = v;
                    }
                    for (k, &(lo, hi)) in id_runs.iter().enumerate() {
                        b.s[(aa + AA_RUNS) as usize + 2 * k] = lo;
                        b.s[(aa + AA_RUNS) as usize + 2 * k + 1] = hi;
                    }
                    let c = &ab.ability_value_with_usage;
                    let (fixed, is_fixed, table) = value_tables(race, i, c);
                    let (vt, vtn) = b.table(&table);
                    for (k, v) in [
                        (PA_PRESENT, 1),
                        (PA_TYPE, ab.ability_type as u32),
                        (PA_USAGE, c.usage as u32),
                        (PA_BASE, f(c.ability_value_base)),
                        (PA_FIXED_VALUE, f(fixed)),
                        (PA_IS_FIXED, is_fixed as u32),
                        (PA_TABLE, vt),
                        (PA_TABLE_COUNT, vtn),
                        (PA_TARGET_TYPE, ab.target.target_type as u32),
                        (PA_TARGET_VALUE, ab.target.value_rh as u32),
                        (PA_TARGET_IS_DEBUFF, ab.target.is_debuff as u32),
                        (PA_AA_TYPE, ab.additional_activate_type as u32),
                        (PA_AA_KIND, aa_kind),
                        (PA_AA_MAX, aa_max as u32),
                        (PA_STATE, aa),
                    ] {
                        b.p[pa + k as usize] = v;
                    }
                    assert_eq!(ab.target.owner, i);
                }
                for (k, v) in [
                    (PD_DEFAULT_COOL_DOWN_TIME, f(det.default_cool_down_time)),
                    (PD_TIME_USAGE, t.usage as u32),
                    (PD_TIME_BASE, f(t.ability_time_base)),
                    (PD_TIME_RATE, f(t.ability_time_by_distance_rate)),
                    (PD_IS_TIME_INFINITE, det.is_time_infinite as u32),
                    (PD_IS_NOT_ACTIVATE_SKILL, det.is_not_activate_skill as u32),
                    (PD_IS_USE_DETAIL_COOL_DOWN, det.is_use_skill_detail_cool_down_time as u32),
                    (PD_PRE, pre),
                    (PD_PRE_COUNT, pre_n),
                    (PD_PRE_FLAGS, pre_f),
                    (PD_MAIN, main),
                    (PD_MAIN_COUNT, main_n),
                    (PD_MAIN_FLAGS, main_f),
                    (PD_ABILITY_COUNT, det.abilities.len() as u32),
                    (PD_ABILITIES, abilities),
                    (PD_TIME_TABLE, tt),
                    (PD_TIME_TABLE_COUNT, ttn),
                    (PD_STATE, ad),
                ] {
                    b.p[pd + k as usize] = v;
                }
            }
            for (k, v) in [
                (PS_ID, base.id as u32),
                (PS_PRIORITY, base.priority as u32),
                (PS_RARITY, m.rarity as u32),
                (PS_IS_GENERAL, m.is_general_skill as u32),
                (PS_TIME_MINUS_ONE, (m.details[0].float_ability_time == -1) as u32),
                (PS_IS_SPECIFIC, specific as u32),
                (PS_TAGS, tags_at),
                (PS_TAG_COUNT, tags.len() as u32),
                (PS_DETAIL0, det_at[0]),
                (PS_DETAIL1, det_at[1]),
                (PS_STATE, st),
            ] {
                b.p[ps + k as usize] = v;
            }
        }
    }
    b.p[PR_STATE_WORDS as usize] = b.s.len() as u32;
    SkillPack { program: b.p, state: b.s, labels: b.labels.unwrap_or_default() }
}

/// An ability's modifier handles as runs of consecutive ids (lo words).
fn modifier_runs(handles: &[ModifierHandle]) -> Vec<(u32, u32)> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for m in handles {
        assert!(m.id < 1 << 32, "modifier ids past 2^32");
        let id = m.id as u32;
        match runs.last_mut() {
            Some((_, hi)) if *hi + 1 == id => *hi = id,
            _ => runs.push((id, id)),
        }
    }
    runs
}

/// (value fixed at construction, whether it is, threshold table of the usage read per activation).
fn value_tables(race: &Race, owner: usize, c: &value::AbilityValue) -> (f32, bool, Vec<(f32, f32)>) {
    let p = &race.data.param.Skill;
    let h = &race.horses[owner];
    let base = c.ability_value_base;
    let lookup = |arr: &[(f32, f32)], x: f32| arr.iter().find(|e| x < e.0).map_or(1.0, |e| e.1);
    let members = |team: i32| race.field.teams.iter().find(|t| t.team_id == team).map_or(&[][..], |t| &t.members[..]);
    match c.usage {
        2 => {
            let sm = &h.skill_manager;
            let mut n = (0..sm.skills.len()).filter(|&s| !sm.is_silent(s)).count() as i32 - 1;
            if n <= 0 {
                n = 0;
            }
            let m = n as f32 * 0.01 + 1.0;
            (if 1.2 > m { m } else { 1.2 } * base, true, vec![])
        }
        3..=7 => {
            let team = h.horse_data.team_id;
            let own = [owner];
            let ms = if team == 0 { &own[..] } else { members(team) };
            let mut sum = 0.0f32;
            for &m in ms {
                let rp = &race.horses[m].horse_data.race_param;
                sum = sum + match c.usage {
                    3 => rp.base_speed,
                    4 => rp.base_stamina,
                    5 => rp.base_pow,
                    6 => rp.base_guts,
                    _ => rp.base_wiz,
                };
            }
            (lookup(&asc(&p.TeamTotalStatusCoefAscArray, |e| (e.StatusThreshold, e.Multiply)), sum) * base, true, vec![])
        }
        13 => {
            let t = asc(&p.MaximumRawStatusCoefAscArray, |e| (e.BaseStatusThreshold, e.Multiply));
            (lookup(&t, c.maximum_raw_status as f32) * base, true, vec![])
        }
        17 => {
            let team = h.horse_data.team_id;
            let m = if team == 0 {
                1.0
            } else {
                let n = members(team).len() as i32;
                if n < 1 { 1.0 } else { 1.0 / n as f32 }
            };
            (m * base, true, vec![])
        }
        18 => {
            let t = asc(&p.BaseWizCoefAscArray, |e| (e.BaseStatusThreshold, e.Multiply));
            (lookup(&t, crate::sim::physics::stats::base_wiz_for(h, crate::api::wiz_mechanic::SKILLS)) * base, true, vec![])
        }
        8 | 9 | 15 | 16 => (0.0, false, vec![]),
        11 => (0.0, false, asc(&p.OrderUpCornerPhaseEndAfterCoefAscArray, |e| (e.OrderChangeCountThreshold as f32, e.Multiply))),
        14 => (
            0.0,
            false,
            asc(&p.ActivateSpecificTagGroupSkillCountCoefAscArray, |e| (e.ActivateSpecificTagGroupSkillCountThreshold as f32, e.Multiply)),
        ),
        // The add values, level-scaled; fixed_value: the add when no threshold matches.
        19 => {
            let lv = |add: f32| value::apply_level_to_value(race.data, c.level, c.ability_type, add, c.level_usage);
            let t = p.DistanceDiffTopCoefAscArray.iter().map(|e| (e.DistanceDiffTopThreshold as f32, lv(e.AddValue as f32 * 0.0001))).collect();
            (lv(0.0), false, t)
        }
        20 => (0.0, false, asc(&p.BlockedSideMaxContinueTimePhaseMiddleRunAbilityValueCoefAscArray1, |e| (e.BlockedSideMaxContinueTimeThreshold as f32, e.Multiply))),
        21 => (0.0, false, asc(&p.BlockedSideMaxContinueTimePhaseMiddleRunAbilityValueCoefAscArray2, |e| (e.BlockedSideMaxContinueTimeThreshold as f32, e.Multiply))),
        22 => (0.0, false, asc(&p.SpeedAbilityValueCoefAscArray1, |e| (e.SpeedThreshold as f32, e.Multiply))),
        23 => (0.0, false, asc(&p.SpeedAbilityValueCoefAscArray2, |e| (e.SpeedThreshold as f32, e.Multiply))),
        // Direct, career win count, fan count, unknown.
        _ => (c.ability_value, true, vec![]),
    }
}
