//! Skill abilities: an ability's effect applied as modifiers on its targets, and additional activations.

use crate::sim::defs::SkillModifierParam;
use crate::sim::race::Race;
use crate::sim::skill::SkillDetailRef;
use crate::sim::skill::modifier::{SkillParamModifier, SkillParamModifierKind};
use crate::sim::skill::target::{self, SkillTarget};
use crate::sim::skill::value::{self, AbilityValue};
use crate::data::SkillAbilityData;

#[derive(Clone, Copy, Debug)]
pub struct ModifierHandle {
    pub horse: usize,
    pub param: SkillModifierParam,
    pub id: u64,
}

/// How an ability earns extra activations after its skill fired. The counters are never reset (whole race).
#[derive(Clone, Copy, Debug, Default)]
pub enum SkillAbilityAdditionalActivate {
    /// Type 0 / unknown: none.
    #[default]
    Null,
    /// Type 1: one per frame the horse gains a position.
    OrderUp { additional_activate_max_count: i32, additional_activate_count: i32 },
    /// Types 2, 3: one per other skill detail the horse activated since the last update.
    ActivateAnySkill { additional_activate_max_count: i32, additional_activate_count: i32 },
}

/// One effect slot of a skill detail (empty slots are `None` in `SkillDetail::abilities`).
#[derive(Clone, Debug, Default)]
pub struct SkillAbility {
    pub target: SkillTarget,
    pub ability_value_with_usage: AbilityValue,
    pub ability_additional_activate: SkillAbilityAdditionalActivate,
    pub level: i32,
    pub additional_activate_type: i32,
    pub modifiers: Vec<ModifierHandle>,
    pub additional_activate_count: i32,
    pub ability_type: i32,
    pub ability_value_on_activate: f32,
}

fn additional_activate_max_count(race: &Race, t: i32) -> i32 {
    race.data.param.Skill.AdditionalActivateAbilityMaxCountArray.iter().find(|x| x.AdditionalActivateType == t).map_or(0, |x| x.AdditionalActivateMaxCount)
}

impl SkillAbility {
    /// No RNG calls: random values draw per activation.
    pub fn new(race: &mut Race, r: SkillDetailRef, level: i32, m: &SkillAbilityData, target: SkillTarget) -> SkillAbility {
        let value = m.float_ability_value as f32 * 0.0001;
        let v = value::apply_level_to_value(race.data, level, m.ability_type, value, m.ability_value_level_usage);
        let calc = value::AbilityValue::new(race, r.horse, m.ability_value_usage, v, level, m.ability_type, m.ability_value_level_usage);
        let t = m.additional_activate_type;
        let ability_additional_activate = match t {
            1 => SkillAbilityAdditionalActivate::OrderUp { additional_activate_max_count: additional_activate_max_count(race, 1), additional_activate_count: 0 },
            2 | 3 => SkillAbilityAdditionalActivate::ActivateAnySkill { additional_activate_max_count: additional_activate_max_count(race, t), additional_activate_count: 0 },
            _ => SkillAbilityAdditionalActivate::Null,
        };
        SkillAbility {
            target,
            ability_value_with_usage: calc,
            ability_additional_activate,
            level,
            additional_activate_type: t,
            modifiers: Vec::new(),
            additional_activate_count: 0,
            ability_type: m.ability_type,
            ability_value_on_activate: 0.0,
        }
    }
}

fn is_muted(race: &Race, r: SkillDetailRef) -> bool {
    let muted = &race.input.horses[r.horse].muted_skills;
    !muted.is_empty() && muted.contains(&race.horses[r.horse].skill_manager.skills[r.skill].id)
}

fn ability_mut<'r>(race: &'r mut Race, r: SkillDetailRef, ability: usize) -> &'r mut SkillAbility {
    race.horses[r.horse].skill_manager.skills[r.skill].details[r.detail].abilities[ability].as_mut().expect("empty ability slot has no state")
}

fn push_modifier(race: &mut Race, r: SkillDetailRef, ability: usize, h: usize, t: SkillModifierParam, mk: impl FnOnce(u64) -> SkillParamModifier) {
    let id = race.next_modifier_id;
    race.next_modifier_id += 1;
    race.horses[h].skill_modifier_receiver.add_modifier(t, mk(id));
    ability_mut(race, r, ability).modifiers.push(ModifierHandle { horse: h, param: t, id });
}

fn add_param(race: &mut Race, r: SkillDetailRef, ability: usize, h: usize, t: SkillModifierParam, v: f32, time: f32) {
    push_modifier(race, r, ability, h, t, |id| SkillParamModifier::new_timed(id, SkillParamModifierKind::Add, v, time));
}

/// Also used for bool params, with `v` = 1.0.
fn set_param(race: &mut Race, r: SkillDetailRef, ability: usize, h: usize, t: SkillModifierParam, v: f32, time: f32) {
    push_modifier(race, r, ability, h, t, |id| SkillParamModifier::new_timed(id, SkillParamModifierKind::Set, v, time));
}

fn add_param_one_time(race: &mut Race, r: SkillDetailRef, ability: usize, h: usize, t: SkillModifierParam, v: f32) {
    push_modifier(race, r, ability, h, t, |id| SkillParamModifier::new_one_time(id, SkillParamModifierKind::Add, v));
}

fn set_param_one_time(race: &mut Race, r: SkillDetailRef, ability: usize, h: usize, t: SkillModifierParam, v: f32) {
    push_modifier(race, r, ability, h, t, |id| SkillParamModifier::new_one_time(id, SkillParamModifierKind::Set, v));
}

/// Order: targets (may draw RNG), value (may draw RNG), the additional-activate gate, then the effects.
///
/// Muted skills (`HorseInput::muted_skills`, an analysis hook) stop right before the effects: no modifier is pushed
/// into any receiver. Everything before still runs: targets and value draws (the only RNG here),
/// `ability_value_on_activate` (so heal counts still see a heal) and the additional-activate counters. Durations,
/// cooldowns and events live in `skill::detail_*` and are untouched; `add_ability_time`/`stop` just walk an empty list,
/// and the skipped modifier ids don't matter (only compared for equality). Exception: types 36/37 (activate random
/// skills; 564 Escapades holds type 37) are not muted, because their modifier feeds an RNG draw
/// (`update_random_activate_skill_id_list`) and muting them would reshuffle the race.
pub fn activate(race: &mut Race, r: SkillDetailRef, ability: usize, is_additional_activate: bool) {
    target::calc_targets(race, r, ability);
    activate_after_targets(race, r, ability, is_additional_activate);
}

/// Split from `activate` so it can be unit-tested with fixed targets.
fn activate_after_targets(race: &mut Race, r: SkillDetailRef, ability: usize, is_additional_activate: bool) {
    let v = value::ability_value(race, r, ability);
    let ab = ability_mut(race, r, ability);
    ab.ability_value_on_activate = v;
    if ab.additional_activate_type == 0 {
        if is_additional_activate {
            ab.additional_activate_count += 1;
        }
    } else {
        if !is_additional_activate {
            return;
        }
        ab.additional_activate_count += 1;
    }
    let ability_type = ab.ability_type;
    if !matches!(ability_type, 36 | 37) && is_muted(race, r) {
        return;
    }
    let ab = ability_mut(race, r, ability);
    let targets = ab.target.targets.clone();
    // The detail's duration, set just before its abilities activate.
    let time = race.horses[r.horse].skill_manager.skills[r.skill].details[r.detail].left_time;
    use SkillModifierParam as P;
    for h in targets {
        match ability_type {
            1 => add_param(race, r, ability, h, P::Speed, v, time),
            2 => add_param(race, r, ability, h, P::Stamina, v, time),
            3 => add_param(race, r, ability, h, P::Power, v, time),
            4 => add_param(race, r, ability, h, P::Guts, v, time),
            5 => add_param(race, r, ability, h, P::Wiz, v, time),
            // Never built (type 6 is an empty slot in the game), kept for completeness.
            6 => set_param(race, r, ability, h, P::RunningStyleExOonige, 1.0, time),
            7 => set_param(race, r, ability, h, P::HpDecRate, v, time),
            8 => add_param(race, r, ability, h, P::VisibleDistance, v, time),
            // HP heal/drain: the target's max HP times v, once.
            9 => {
                let hp = race.horses[h].max_hp * v;
                add_param_one_time(race, r, ability, h, P::Hp, hp);
            }
            10 => set_param_one_time(race, r, ability, h, P::StartDelayScale, if v < 0.0 { 0.0 } else { v }),
            11 => set_param(race, r, ability, h, P::ForceOvertakeIn, 1.0, time),
            12 => set_param(race, r, ability, h, P::ForceOvertakeOut, 1.0, time),
            13 => set_param_one_time(race, r, ability, h, P::TemptationEndTime, v),
            // The higher StartDelayFix wins: a larger existing value stays, a smaller one is replaced.
            14 => {
                let rec = &mut race.horses[h].skill_modifier_receiver;
                if rec.has_modifier(P::StartDelayFix) {
                    if rec.apply_modifier(P::StartDelayFix, 0.0) > v {
                        continue;
                    }
                    rec.remove_modifier(P::StartDelayFix);
                }
                set_param_one_time(race, r, ability, h, P::StartDelayFix, v);
            }
            21 => add_param(race, r, ability, h, P::CurrentSpeed, v, time),
            22 => add_param(race, r, ability, h, P::CurrentSpeedWithNaturalDeceleration, v, time),
            27 => add_param(race, r, ability, h, P::TargetSpeed, v, time),
            28 => {
                add_param(race, r, ability, h, P::LaneMoveSpeed, v, time);
                if 0.0 < v {
                    set_param(race, r, ability, h, P::TargetSpeedOnMoveLane, 1.0, time);
                }
            }
            29 => add_param(race, r, ability, h, P::TemptationPer, v, time),
            30 => add_param(race, r, ability, h, P::PushPer, v, time),
            31 => add_param(race, r, ability, h, P::Accel, v, time),
            32 => {
                add_param(race, r, ability, h, P::Speed, v, time);
                add_param(race, r, ability, h, P::Stamina, v, time);
                add_param(race, r, ability, h, P::Power, v, time);
                add_param(race, r, ability, h, P::Guts, v, time);
                add_param(race, r, ability, h, P::Wiz, v, time);
            }
            35 => add_param(race, r, ability, h, P::TargetLane, v, time),
            36 => set_param_one_time(race, r, ability, h, P::ActivateRandomNormalAndRareSkill, v),
            37 => set_param_one_time(race, r, ability, h, P::ActivateRandomRareSkill, v),
            38 => set_param(race, r, ability, h, P::DebuffCancel, 1.0, time),
            _ => {}
        }
    }
}

/// The pushed modifiers expire on their own timers; only the handles are dropped.
pub fn stop(race: &mut Race, r: SkillDetailRef, ability: usize) {
    let ab = ability_mut(race, r, ability);
    ab.modifiers.clear();
    ab.additional_activate_count = 0;
}

pub fn add_ability_time(race: &mut Race, r: SkillDetailRef, ability: usize, additional_ability_time: f32) {
    let handles = std::mem::take(&mut ability_mut(race, r, ability).modifiers);
    for m in &handles {
        race.horses[m.horse].skill_modifier_receiver.add_ability_time(m.param, m.id, additional_ability_time);
    }
    ability_mut(race, r, ability).modifiers = handles;
}

/// Additional activations earned this frame (counted against the cap).
pub fn calc_additional_activate_count(race: &mut Race, r: SkillDetailRef, ability: usize) -> i32 {
    let owner = &race.horses[r.horse];
    let gained_position = owner.order_change_counter_up.is_order_change_per_frame;
    let skill_id = owner.skill_manager.skills[r.skill].skill_master.id;
    let prev = &owner.skill_manager.prev_activated_details;
    let prev_count = prev.len() as i32;
    let prev_has_self = prev.iter().any(|d| race.horses[d.horse].skill_manager.skills[d.skill].skill_master.id == skill_id);
    match &mut ability_mut(race, r, ability).ability_additional_activate {
        SkillAbilityAdditionalActivate::Null => 0,
        SkillAbilityAdditionalActivate::OrderUp { additional_activate_max_count: max, additional_activate_count: count } => {
            if gained_position && *count < *max {
                *count += 1;
                1
            } else {
                0
            }
        }
        SkillAbilityAdditionalActivate::ActivateAnySkill { additional_activate_max_count: max, additional_activate_count: count } => {
            let mut remain = *max - *count;
            if remain < 1 || prev_count < 1 {
                return 0;
            }
            let n = if prev_has_self { prev_count - 1 } else { prev_count };
            if n < remain {
                remain = n;
            }
            *count += remain;
            remain
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::test_util::{input, skill_race};
    use SkillModifierParam as P;

    const R: SkillDetailRef = SkillDetailRef { horse: 0, skill: 0, detail: 0 };

    fn ab_data(ability_type: i32, value: i32, usage: i32, additional: i32) -> SkillAbilityData {
        SkillAbilityData { ability_type, ability_value_usage: usage, additional_activate_type: additional, ability_value_level_usage: 1, float_ability_value: value, target_type: 1, target_value: 0 }
    }

    fn install(race: &mut Race, m: &SkillAbilityData, level: i32, targets: Vec<usize>, left_time: f32) {
        let target = SkillTarget { target_type: 1, targets, ..Default::default() };
        let ab = SkillAbility::new(race, R, level, m, target);
        let d = &mut race.horses[0].skill_manager.skills[0].details[0];
        d.abilities = vec![Some(ab)];
        d.left_time = left_time;
    }

    fn ab<'r>(race: &'r Race) -> &'r SkillAbility {
        race.horses[0].skill_manager.skills[0].details[0].abilities[0].as_ref().unwrap()
    }

    #[test]
    fn construction() {
        let inp = input(1);
        let mut race = skill_race(&inp, 1);
        // TargetSpeed 0.35 at level 5, level usage Normal: 1.1f * 0.35f = 0.385f
        install(&mut race, &ab_data(27, 3500, 1, 0), 5, vec![0], 2.4);
        assert_eq!(ab(&race).ability_value_with_usage.ability_value, 0.385f32);
        assert!(matches!(ab(&race).ability_additional_activate, SkillAbilityAdditionalActivate::Null));
        for (t, max) in [(1, 3), (2, 3), (3, 2)] {
            install(&mut race, &ab_data(27, 3500, 1, t), 1, vec![0], 2.4);
            let got = match ab(&race).ability_additional_activate {
                SkillAbilityAdditionalActivate::OrderUp { additional_activate_max_count, .. } if t == 1 => additional_activate_max_count,
                SkillAbilityAdditionalActivate::ActivateAnySkill { additional_activate_max_count, .. } if t != 1 => additional_activate_max_count,
                _ => panic!("wrong additional-activate class for type {t}"),
            };
            assert_eq!(got, max);
        }
    }

    #[test]
    fn effects_time_and_stop() {
        let inp = input(2);
        let mut race = skill_race(&inp, 1);
        install(&mut race, &ab_data(27, 3500, 1, 0), 1, vec![0, 1], 2.4);
        activate_after_targets(&mut race, R, 0, false);
        assert_eq!(ab(&race).ability_value_on_activate, 0.35f32);
        for h in 0..2 {
            assert_eq!(race.horses[h].skill_modifier_receiver.apply_modifier(P::TargetSpeed, 0.0), 0.35f32);
        }
        assert_eq!(ab(&race).modifiers.len(), 2);
        assert_eq!(ab(&race).additional_activate_count, 0);
        // add_ability_time extends every pushed modifier; stop forgets them (they stay in the receivers).
        add_ability_time(&mut race, R, 0, 1.0);
        let m = race.horses[1].skill_modifier_receiver.modifier[0].1[0];
        assert_eq!(m.time, 1.0f32 + 2.4f32);
        stop(&mut race, R, 0);
        assert!(ab(&race).modifiers.is_empty());
        assert!(race.horses[1].skill_modifier_receiver.has_modifier(P::TargetSpeed));
        // Additional-type abilities: nothing on the normal activation (value still evaluated), applied on an additional activation.
        install(&mut race, &ab_data(31, 2000, 1, 1), 1, vec![0], 3.0);
        activate_after_targets(&mut race, R, 0, false);
        assert!(!race.horses[0].skill_modifier_receiver.has_modifier(P::Accel));
        activate_after_targets(&mut race, R, 0, true);
        assert_eq!(race.horses[0].skill_modifier_receiver.apply_modifier(P::Accel, 0.0), 0.19999999f32); // (f32)2000 * 0.0001f (numpy)
        assert_eq!(ab(&race).additional_activate_count, 1);
    }

    #[test]
    fn special_types() {
        let inp = input(1);
        let mut race = skill_race(&inp, 1);
        let rec = |race: &Race| race.horses[0].skill_modifier_receiver.clone();
        // HpRate: maxHp * v, one-time Add
        race.horses[0].max_hp = 1000.0;
        install(&mut race, &ab_data(9, 150, 1, 0), 1, vec![0], 0.0);
        activate_after_targets(&mut race, R, 0, false);
        assert_eq!(rec(&race).apply_modifier(P::Hp, 0.0), 1000.0f32 * (150 as f32 * 0.0001));
        assert!(rec(&race).modifier.iter().find(|(k, _)| *k == P::Hp).unwrap().1[0].is_one_time);
        // StartDash: negative clamped to 0, one-time Set
        install(&mut race, &ab_data(10, -5000, 1, 0), 1, vec![0], 0.0);
        activate_after_targets(&mut race, R, 0, false);
        assert_eq!(rec(&race).apply_modifier(P::StartDelayScale, 1.0), 0.0);
        // StartDelayFix "higher": keep the larger value
        install(&mut race, &ab_data(14, 5000, 1, 0), 1, vec![0], 0.0);
        activate_after_targets(&mut race, R, 0, false);
        install(&mut race, &ab_data(14, 3000, 1, 0), 1, vec![0], 0.0);
        activate_after_targets(&mut race, R, 0, false);
        assert!(ab(&race).modifiers.is_empty()); // early return: nothing pushed
        assert_eq!(rec(&race).apply_modifier(P::StartDelayFix, 0.0), 0.5f32);
        install(&mut race, &ab_data(14, 8000, 1, 0), 1, vec![0], 0.0);
        activate_after_targets(&mut race, R, 0, false);
        let r = rec(&race);
        assert_eq!(r.apply_modifier(P::StartDelayFix, 0.0), 0.79999995f32); // (f32)8000 * 0.0001f
        assert_eq!(r.modifier.iter().find(|(k, _)| *k == P::StartDelayFix).unwrap().1.len(), 1);
        // LaneMoveSpeed > 0 also sets TargetSpeedOnMoveLane; DebuffCancel-style bools are Set 1.0
        install(&mut race, &ab_data(28, 200, 1, 0), 1, vec![0], 1.5);
        activate_after_targets(&mut race, R, 0, false);
        assert!(rec(&race).has_modifier(P::TargetSpeedOnMoveLane));
        install(&mut race, &ab_data(38, 10000, 1, 0), 1, vec![0], 1.5);
        activate_after_targets(&mut race, R, 0, false);
        assert!(rec(&race).apply_modifier_bool(P::DebuffCancel, false));
    }

    #[test]
    fn additional_activate_counts() {
        let inp = input(2);
        let mut race = skill_race(&inp, 1);
        race.horses[0].skill_manager.skills[0].skill_master.id = 100;
        race.horses[0].skill_manager.skills.push(Default::default());
        race.horses[0].skill_manager.skills[1].skill_master.id = 200;
        // OrderUp: 1 per order-up frame, max 3
        install(&mut race, &ab_data(27, 3500, 1, 1), 1, vec![0], 2.0);
        assert_eq!(calc_additional_activate_count(&mut race, R, 0), 0);
        race.horses[0].order_change_counter_up.is_order_change_per_frame = true;
        let n: i32 = (0..5).map(|_| calc_additional_activate_count(&mut race, R, 0)).sum();
        assert_eq!(n, 3);
        // ActivateAnySkill (type 3, max 2): other recently activated details, minus one if this skill is among them
        install(&mut race, &ab_data(27, 3500, 1, 3), 1, vec![0], 2.0);
        let sr = |skill| SkillDetailRef { horse: 0, skill, detail: 0 };
        race.horses[0].skill_manager.prev_activated_details = vec![sr(0)];
        assert_eq!(calc_additional_activate_count(&mut race, R, 0), 0);
        race.horses[0].skill_manager.prev_activated_details = vec![sr(0), sr(1)];
        assert_eq!(calc_additional_activate_count(&mut race, R, 0), 1);
        race.horses[0].skill_manager.prev_activated_details = vec![sr(1), sr(1), sr(1)];
        assert_eq!(calc_additional_activate_count(&mut race, R, 0), 1); // capped by max - count
        assert_eq!(calc_additional_activate_count(&mut race, R, 0), 0);
    }

}
