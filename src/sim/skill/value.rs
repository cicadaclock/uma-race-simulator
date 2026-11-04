//! Ability values and durations: the value/time calculators, level scaling, and course-distance scaling of
//! durations and cooldowns.
//!
//! Each calculator family is one struct dispatched on its usage id (only the fields a usage needs are meaningful).
//! A calculator's owner is always the skill owner (`SkillDetailRef::horse`).

use crate::sim::race::Race;
use crate::sim::field::first_horse;
use crate::sim::skill::SkillDetailRef;
use crate::sim::util;
use crate::data::GameData;

/// How an ability's value is computed. `usage`: 0/1/unknown direct, 2 skill count, 3–7 team total status,
/// 8/9 random, 10 career win count, 11 positions gained (`change_count_corner_phase_end_after`), 12 fan count,
/// 13 maximum raw status, 14 tag-group skill activations, 15 heal skill activations, 16 final-corner-end order,
/// 17 inverse team size, 18 base wiz, 19 add by distance behind the leader, 20/21 blocked-side time, 22/23 speed.
#[derive(Clone, Debug, Default)]
pub struct AbilityValue {
    pub usage: i32,
    /// Already level-scaled.
    pub ability_value_base: f32,
    /// Usages fixed at construction (direct, win count, fan count): the final value.
    pub ability_value: f32,
    /// Usage 13: read at construction.
    pub maximum_raw_status: i32,
    /// Usage 19 scales its added value by level like the base value: level, ability type, level usage.
    pub level: i32,
    pub ability_type: i32,
    pub level_usage: i32,
}

/// How a skill detail's duration is computed. `usage`: 0/1/unknown direct, 2 distance behind the leader,
/// 3/7 remaining HP, 4 extended per position gained, 5/6 blocked-side time.
#[derive(Clone, Debug, Default)]
pub struct AbilityTime {
    pub usage: i32,
    pub ability_time_base: f32,
    /// Usage 4.
    pub ability_time_by_distance_rate: f32,
    /// Usage 4.
    pub increment_count: i32,
    /// Usage 4; 0 for every other usage.
    pub additional_ability_time: f32,
}

/// Ascending threshold table: the first entry with `x < threshold`, else 1.0.
fn asc_lookup<T: PartialOrd + Copy>(arr: impl IntoIterator<Item = (T, f32)>, x: T) -> f32 {
    for (threshold, multiply) in arr {
        if x < threshold {
            return multiply;
        }
    }
    1.0
}

fn members<'r>(race: &'r Race, team_id: i32) -> &'r [usize] {
    race.field.teams.iter().find(|t| t.team_id == team_id).map_or(&[], |t| &t.members)
}

impl AbilityValue {
    /// No RNG calls (random usages draw per read). Unknown usages fall back to direct.
    pub fn new(race: &mut Race, owner: usize, usage: i32, ability_value_base: f32, level: i32, ability_type: i32, level_usage: i32) -> AbilityValue {
        let p = &race.data.param.Skill;
        let mut c = AbilityValue { usage, ability_value_base, level, ability_type, level_usage, ..Default::default() };
        match usage {
            10 => {
                let n = race.input.horses[owner].single_mode_win_count;
                c.ability_value = asc_lookup(p.RaceWinCountCoefAscArray.iter().map(|e| (e.WinCountThreshold, e.Multiply)), n) * ability_value_base;
            }
            12 => {
                let n = race.input.horses[owner].fan_count;
                c.ability_value = asc_lookup(p.FanCountCoefAscArray.iter().map(|e| (e.FanCountThreshold, e.Multiply)), n) * ability_value_base;
            }
            13 => {
                let rp = &race.horses[owner].horse_data.race_param;
                c.maximum_raw_status = [rp.raw_speed, rp.raw_guts, rp.raw_pow, rp.raw_stamina, rp.raw_wiz].into_iter().max().unwrap();
            }
            2..=9 | 11 | 14..=23 => {}
            _ => c.ability_value = ability_value_base,
        }
        c
    }
}

/// The ability's value, read once per activation. Usages 8/9 draw one `random_int` per call.
pub fn ability_value(race: &mut Race, r: SkillDetailRef, ability: usize) -> f32 {
    let c = race.horses[r.horse].skill_manager.skills[r.skill].details[r.detail].abilities[ability]
        .as_ref()
        .expect("ability_value on an empty ability slot")
        .ability_value_with_usage
        .clone();
    calc_ability_value(race, r.horse, &c)
}

fn calc_ability_value(race: &mut Race, owner: usize, c: &AbilityValue) -> f32 {
    let data = race.data;
    let p = &data.param.Skill;
    let base = c.ability_value_base;
    let h = &race.horses[owner];
    let multiply = match c.usage {
        2 => {
            let mut n = (0..h.skill_manager.skills.len()).filter(|&s| !h.skill_manager.is_silent(s)).count() as i32 - 1;
            if n <= 0 {
                n = 0;
            }
            let m = n as f32 * 0.01 + 1.0;
            if 1.2 > m { m } else { 1.2 }
        }
        // Sum of the team's base stat (just the owner without a team).
        3..=7 => {
            let team = h.horse_data.team_id;
            let self_only = [owner];
            let members = if team == 0 { &self_only[..] } else { members(race, team) };
            let mut sum = 0.0f32;
            for &m in members {
                let rp = &race.horses[m].horse_data.race_param;
                sum = sum + match c.usage {
                    3 => rp.base_speed,
                    4 => rp.base_stamina,
                    5 => rp.base_pow,
                    6 => rp.base_guts,
                    _ => rp.base_wiz,
                };
            }
            asc_lookup(p.TeamTotalStatusCoefAscArray.iter().map(|e| (e.StatusThreshold, e.Multiply)), sum)
        }
        // Weighted lot.
        8 | 9 => {
            let arr: Vec<(f32, i32)> = if c.usage == 8 {
                p.AbiiltyValueUsageRandom1Array.iter().map(|e| (e.AbilityValueCoef, e.Per)).collect()
            } else {
                p.AbiiltyValueUsageRandom2Array.iter().map(|e| (e.AbilityValueCoef, e.Per)).collect()
            };
            let total: i32 = arr.iter().map(|e| e.1).sum();
            let mut v = race.rng.random_int(total);
            let mut res = 1.0;
            for (coef, per) in arr {
                v -= per;
                if v < 1 {
                    res = coef;
                    break;
                }
            }
            res
        }
        10 | 12 => return c.ability_value,
        11 => {
            let n = h.order_change_counter_up.change_count_corner_phase_end_after;
            asc_lookup(p.OrderUpCornerPhaseEndAfterCoefAscArray.iter().map(|e| (e.OrderChangeCountThreshold, e.Multiply)), n)
        }
        13 => asc_lookup(p.MaximumRawStatusCoefAscArray.iter().map(|e| (e.BaseStatusThreshold, e.Multiply)), c.maximum_raw_status as f32),
        14 => {
            let n = h.activate_specific_tag_group_skill_count;
            asc_lookup(p.ActivateSpecificTagGroupSkillCountCoefAscArray.iter().map(|e| (e.ActivateSpecificTagGroupSkillCountThreshold, e.Multiply)), n)
        }
        15 => util::min(h.activate_heal_skill_count as f32 * 0.1 + 1.0, 1.5),
        16 => {
            if h.final_corner_end_order == -1 {
                1.0
            } else {
                util::max(1.2 - h.final_corner_end_order as f32 * 0.02, 1.0)
            }
        }
        17 => {
            let team = h.horse_data.team_id;
            if team == 0 {
                1.0
            } else {
                let n = members(race, team).len() as i32;
                if n < 1 { 1.0 } else { 1.0 / n as f32 }
            }
        }
        18 => asc_lookup(p.BaseWizCoefAscArray.iter().map(|e| (e.BaseStatusThreshold, e.Multiply)), crate::sim::physics::stats::base_wiz_for(h, crate::api::wiz_mechanic::SKILLS)),
        // Adds a level-scaled bonus by distance behind the leader to the base value.
        19 => {
            let diff = race.horses[first_horse(race)].distance - h.distance;
            let mut add = 0.0f32;
            for e in &p.DistanceDiffTopCoefAscArray {
                if diff < e.DistanceDiffTopThreshold as f32 {
                    add = e.AddValue as f32 * 0.0001;
                    break;
                }
            }
            return apply_level_to_value(data, c.level, c.ability_type, add, c.level_usage) + base;
        }
        // Longest side-block time in the middle phase.
        20 | 21 => {
            let arr = if c.usage == 20 {
                &p.BlockedSideMaxContinueTimePhaseMiddleRunAbilityValueCoefAscArray1
            } else {
                &p.BlockedSideMaxContinueTimePhaseMiddleRunAbilityValueCoefAscArray2
            };
            let t = h.block_side_max_continue_time_by_phase[1];
            asc_lookup(arr.iter().map(|e| (e.BlockedSideMaxContinueTimeThreshold as f32, e.Multiply)), t)
        }
        // Skill-modified speed stat.
        22 | 23 => {
            let s = crate::sim::physics::stats::speed(h);
            let arr: Vec<(f32, f32)> = if c.usage == 22 {
                p.SpeedAbilityValueCoefAscArray1.iter().map(|e| (e.SpeedThreshold as f32, e.Multiply)).collect()
            } else {
                p.SpeedAbilityValueCoefAscArray2.iter().map(|e| (e.SpeedThreshold as f32, e.Multiply)).collect()
            };
            asc_lookup(arr, s)
        }
        _ => return c.ability_value,
    };
    multiply * base
}

/// Scales `value` by the master `skill_level_value` coefficient for (ability type, level), `(float)raw * 0.0001f`,
/// 1.0 without a row. Level usage 1 multiplies, 3 divides, anything else leaves `value` alone.
pub fn apply_level_to_value(data: &GameData, level: i32, ability_type: i32, value: f32, level_usage: i32) -> f32 {
    // Rows are in id order, so the first match is the one a stable sort by level would pick.
    let coef = data
        .master
        .skill_level_value
        .iter()
        .find(|e| e.ability_type == ability_type && e.level == level)
        .map_or(1.0, |e| e.float_ability_value_coef as f32 * 0.0001);
    match level_usage {
        1 => coef * value,
        3 => (1.0 / coef) * value,
        _ => value,
    }
}

impl AbilityTime {
    /// No RNG calls.
    pub fn new(usage: i32, ability_time_base: f32, ability_time_by_distance_rate: f32) -> AbilityTime {
        AbilityTime { usage, ability_time_base, ability_time_by_distance_rate, increment_count: 0, additional_ability_time: 0.0 }
    }
}

fn time_calc<'r>(race: &'r Race, r: SkillDetailRef) -> &'r AbilityTime {
    &race.horses[r.horse].skill_manager.skills[r.skill].details[r.detail].time
}

fn time_calc_mut<'r>(race: &'r mut Race, r: SkillDetailRef) -> &'r mut AbilityTime {
    &mut race.horses[r.horse].skill_manager.skills[r.skill].details[r.detail].time
}

/// The detail's duration, read once when it activates.
pub fn ability_time(race: &Race, r: SkillDetailRef) -> f32 {
    let c = time_calc(race, r);
    let p = &race.data.param.Skill;
    let h = &race.horses[r.horse];
    let multiply = match c.usage {
        2 => {
            let v = (race.horses[first_horse(race)].distance - h.distance) / p.AbilityTimeDiffDistanceTopCoefPerDistance
                + p.AbilityTimeDiffDistanceTopCoefMin;
            util::clamp(v, p.AbilityTimeDiffDistanceTopCoefMin, p.AbilityTimeDiffDistanceTopCoefMax)
        }
        3 => asc_lookup(p.TimeRemainHpCoefAscArray.iter().map(|e| (e.HpThreshold, e.Multiply)), h.hp),
        7 => asc_lookup(p.TimeRemainHpCoefAscArray2.iter().map(|e| (e.HpThreshold, e.Multiply)), h.hp),
        // Longest side-block time in the middle phase.
        5 | 6 => {
            let arr = if c.usage == 5 {
                &p.BlockedSideMaxContinueTimePhaseMiddleRunAbilityTimeCoefAscArray1
            } else {
                &p.BlockedSideMaxContinueTimePhaseMiddleRunAbilityTimeCoefAscArray2
            };
            let t = h.block_side_max_continue_time_by_phase[1];
            asc_lookup(arr.iter().map(|e| (e.BlockedSideMaxContinueTimeThreshold as f32, e.Multiply)), t)
        }
        // Direct and usage 4: the base time.
        _ => return c.ability_time_base,
    };
    multiply * c.ability_time_base
}

/// Extra duration earned this frame (usage 4 only).
pub fn additional_ability_time(race: &Race, r: SkillDetailRef) -> f32 {
    let c = time_calc(race, r);
    if c.usage == 4 { c.additional_ability_time } else { 0.0 }
}

/// Resets the usage-4 state; runs when the detail stops.
pub fn time_calculator_init(race: &mut Race, r: SkillDetailRef) {
    let c = time_calc_mut(race, r);
    if c.usage == 4 {
        c.additional_ability_time = 0.0;
        c.increment_count = 0;
    }
}

/// Usage 4: each frame the horse gains a position adds `OrderUpAddAbilityTime` (distance-scaled), up to a max count.
pub fn update_ability_time(race: &mut Race, r: SkillDetailRef) {
    if time_calc(race, r).usage != 4 {
        return;
    }
    let gained_position = race.horses[r.horse].order_change_counter_up.is_order_change_per_frame;
    let p = &race.data.param.Skill;
    let (max_count, add_time) = (p.OrderUpAddAbilityTimeMaxCount, p.OrderUpAddAbilityTime);
    let c = time_calc_mut(race, r);
    if !gained_position || max_count <= c.increment_count {
        c.additional_ability_time = 0.0;
        return;
    }
    c.increment_count += 1;
    c.additional_ability_time = add_time * c.ability_time_by_distance_rate;
}

pub fn calc_ability_time_by_distance_rate(course_distance: i32, ability_time_divide_distance: i32) -> f32 {
    course_distance as f32 / ability_time_divide_distance as f32
}

pub fn calc_cool_down_time_by_distance(cool_down_time: f32, course_distance: i32, cool_down_time_divide_distance: i32) -> f32 {
    (course_distance as f32 / cool_down_time_divide_distance as f32) * cool_down_time
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::ai::order::Standings;
    use crate::sim::field::HorseTeamInfo;
    use crate::sim::test_util::{data, input, skill_race};

    fn val(race: &mut Race, owner: usize, usage: i32, base: f32) -> f32 {
        let c = AbilityValue::new(race, owner, usage, base, 1, 1, 0);
        calc_ability_value(race, owner, &c)
    }

    #[test]
    fn level_scaling_from_master() {
        let d = data();
        let v = 3500 as f32 * 0.0001; // 0.35f
        // type 27 level 5: coef (f32)11000 * 0.0001f = 1.1f; 1.1f * 0.35f = 0.385f (numpy f32)
        assert_eq!(apply_level_to_value(d, 5, 27, v, 1), 0.385f32);
        assert_eq!(apply_level_to_value(d, 5, 27, v, 3), 0.3181818f32);
        assert_eq!(apply_level_to_value(d, 5, 27, v, 0), v);
        assert_eq!(apply_level_to_value(d, 5, 27, v, 2), v);
        assert_eq!(apply_level_to_value(d, 99, 27, v, 1), v); // no row → coef 1
    }

    #[test]
    fn value_usages() {
        let mut inp = input(3);
        inp.horses[0].single_mode_win_count = 10; // < 14 → 0.9
        inp.horses[0].fan_count = 200000; // ≥ 160000 → 1.2
        let mut race = skill_race(&inp, 1);
        let base = 0.35f32;
        assert_eq!(val(&mut race, 0, 0, base), base);
        assert_eq!(val(&mut race, 0, 1, base), base);
        assert_eq!(val(&mut race, 0, 99, base), base); // unknown → Direct
        // static: win count / fan count
        assert_eq!(val(&mut race, 0, 10, base), 0.9f32 * base);
        assert_eq!(val(&mut race, 0, 12, base), 1.2f32 * base);
        // team total: team 0 = self only; 1100 < 1200 → 0.8
        race.horses[0].horse_data.race_param.base_speed = 1100.0;
        race.horses[1].horse_data.race_param.base_speed = 1000.0;
        assert_eq!(val(&mut race, 0, 3, base), 0.28f32);
        race.horses[0].horse_data.team_id = 1;
        race.field.teams = vec![HorseTeamInfo { team_id: 1, members: vec![0, 1] }];
        assert_eq!(val(&mut race, 0, 3, base), base); // 2100 < 2600 → 1.0
        assert_eq!(val(&mut race, 0, 17, base), 0.5f32 * base);
        race.horses[0].horse_data.team_id = 2; // not in the list → empty array → sum 0 → 0.8
        assert_eq!(val(&mut race, 0, 3, base), 0.28f32);
        assert_eq!(val(&mut race, 0, 17, base), base);
        // order-up count corner phase end-after: 3 → 1.2
        race.horses[0].order_change_counter_up.change_count_corner_phase_end_after = 3;
        assert_eq!(val(&mut race, 0, 11, base), 1.2f32 * base);
        // maximum raw status (read at construction): max 1050 → < 1100 → 1.1
        let rp = &mut race.horses[0].horse_data.race_param;
        (rp.raw_speed, rp.raw_stamina, rp.raw_pow, rp.raw_guts, rp.raw_wiz) = (900, 1050, 700, 300, 1000);
        let c = AbilityValue::new(&mut race, 0, 13, base, 1, 1, 0);
        race.horses[0].horse_data.race_param.raw_stamina = 2000;
        assert_eq!(calc_ability_value(&mut race, 0, &c), 1.1f32 * base);
        // specific tag count: 3 → (3 < 5) 1.0; 0 → 0.0
        race.horses[0].activate_specific_tag_group_skill_count = 3;
        assert_eq!(val(&mut race, 0, 14, base), base);
        race.horses[0].activate_specific_tag_group_skill_count = 0;
        assert_eq!(val(&mut race, 0, 14, base), 0.0);
        // heal count 2: 2 * 0.1f + 1 = 1.2f; 9 → capped 1.5
        race.horses[0].activate_heal_skill_count = 2;
        assert_eq!(val(&mut race, 0, 15, base), 1.2f32 * base);
        race.horses[0].activate_heal_skill_count = 9;
        assert_eq!(val(&mut race, 0, 15, base), 1.5f32 * base);
        // final corner end order: uninitialised → 1.0; 3 → 1.2f - 3*0.02f = 1.1400001f
        assert_eq!(val(&mut race, 0, 16, base), base);
        race.horses[0].final_corner_end_order = 3;
        assert_eq!(val(&mut race, 0, 16, base), 1.1400001f32 * base);
        // skill num: 1 skill → n = 0 → 1.0
        assert_eq!(val(&mut race, 0, 2, base), base);
        // base wiz 900 → < 1000 → 1.1
        race.horses[0].horse_data.race_param.base_wiz = 900.0;
        assert_eq!(val(&mut race, 0, 18, base), 1.1f32 * base);
        // blocked side (phase MiddleRun): 3 s → < 4 → 2.0; array 2 empty → 1.0
        race.horses[0].block_side_max_continue_time_by_phase[1] = 3.0;
        assert_eq!(val(&mut race, 0, 20, base), 2.0 * base);
        assert_eq!(val(&mut race, 0, 21, base), base);
        // distance diff top: leader 25 m ahead → +(f32)1000*0.0001f (level usage 0) + base
        race.field.standings = Standings::new(3);
        race.field.standings.horse_array_by_order.swap(0, 1);
        race.horses[1].distance = 125.0;
        race.horses[0].distance = 100.0;
        assert_eq!(val(&mut race, 0, 19, base), 1000 as f32 * 0.0001 + base);
        race.horses[1].distance = 110.0;
        assert_eq!(val(&mut race, 0, 19, base), 0.0 + base);
        assert_eq!(race.rng.draws(), 0);
    }

    #[test]
    fn multiply_random_draws_once_per_read() {
        let inp = input(1);
        let mut race = skill_race(&inp, 12345);
        let mut reference = crate::rng::RaceRandom::new(12345);
        let base = -1.0f32;
        let c = AbilityValue::new(&mut race, 0, 8, base, 1, 9, 0);
        assert_eq!(race.rng.draws(), 0); // construction draws nothing
        let mut seen = [false; 3];
        for _ in 0..200 {
            // random_int(100); r -= 60 → < 1 ⇒ 0.0; r -= 30 → < 1 ⇒ 0.02; else 0.04 (61/30/9 split)
            let r = reference.random_int(100);
            let (i, coef) = if r - 60 < 1 { (0, 0.0f32) } else if r - 90 < 1 { (1, 0.02) } else { (2, 0.04) };
            seen[i] = true;
            assert_eq!(calc_ability_value(&mut race, 0, &c), coef * base);
            assert_eq!(race.rng.draws(), reference.draws());
        }
        assert_eq!(seen, [true; 3]);
    }

    #[test]
    fn time_usages() {
        let inp = input(2);
        let mut race = skill_race(&inp, 1);
        let r = SkillDetailRef { horse: 0, skill: 0, detail: 0 };
        let set = |race: &mut Race, usage: i32, base: f32, rate: f32| {
            let c = AbilityTime::new(usage, base, rate);
            race.horses[0].skill_manager.skills[0].details[0].time = c;
        };
        set(&mut race, 1, 12.0, 2.4);
        assert_eq!(ability_time(&race, r), 12.0);
        assert_eq!(additional_ability_time(&race, r), 0.0);
        // remain HP: 2500 → < 2600 → 2.0 (array 1); array 2: 2500 → 3.0 (falls to 99999)
        race.horses[0].hp = 2500.0;
        set(&mut race, 3, 12.0, 2.4);
        assert_eq!(ability_time(&race, r), 24.0);
        set(&mut race, 7, 12.0, 2.4);
        assert_eq!(ability_time(&race, r), 36.0);
        // distance diff top: leader (horse 1) at 70, self at 40: (30 / 62.5f) + 0.8f = 1.28f; * 12 = 15.36f (numpy f32)
        race.field.standings = Standings::new(2);
        race.field.standings.horse_array_by_order.swap(0, 1);
        race.horses[1].distance = 70.0;
        race.horses[0].distance = 40.0;
        set(&mut race, 2, 12.0, 2.4);
        assert_eq!(ability_time(&race, r), 15.36f32);
        race.horses[1].distance = 40.0; // diff 0 → 0.8 (min)
        assert_eq!(ability_time(&race, r), 0.8f32 * 12.0);
        race.horses[1].distance = 1000.0; // → max 1.6
        assert_eq!(ability_time(&race, r), 1.6f32 * 12.0);
        // blocked side: 5 s → < 6 → 3.0; array 2 empty → 1.0
        race.horses[0].block_side_max_continue_time_by_phase[1] = 5.0;
        set(&mut race, 5, 12.0, 2.4);
        assert_eq!(ability_time(&race, r), 36.0);
        set(&mut race, 6, 12.0, 2.4);
        assert_eq!(ability_time(&race, r), 12.0);
        // increment order-up: +1.0 * rate per order-up frame, max 3, reset by Init
        set(&mut race, 4, 12.0, 2.4);
        assert_eq!(ability_time(&race, r), 12.0);
        update_ability_time(&mut race, r);
        assert_eq!(additional_ability_time(&race, r), 0.0);
        race.horses[0].order_change_counter_up.is_order_change_per_frame = true;
        for _ in 0..3 {
            update_ability_time(&mut race, r);
            assert_eq!(additional_ability_time(&race, r), 1.0f32 * 2.4f32);
        }
        update_ability_time(&mut race, r);
        assert_eq!(additional_ability_time(&race, r), 0.0);
        time_calculator_init(&mut race, r);
        update_ability_time(&mut race, r);
        assert_eq!(additional_ability_time(&race, r), 2.4f32);
    }

    #[test]
    fn asc_lookup_strict_and_fallthrough() {
        let arr = [(2, 1.0f32), (3, 1.1), (4, 1.2), (99, 1.25)];
        assert_eq!(asc_lookup(arr, 0), 1.0);
        assert_eq!(asc_lookup(arr, 2), 1.1); // strict: 2 < 2 is false
        assert_eq!(asc_lookup(arr, 3), 1.2);
        assert_eq!(asc_lookup(arr, 99), 1.0); // falls off the end → 1.0, not the last entry
        let f = [(1200.0f32, 0.8f32), (1800.0, 0.9)];
        assert_eq!(asc_lookup(f, 1199.99), 0.8);
        assert_eq!(asc_lookup(f, f32::NAN), 1.0);
    }

    #[test]
    fn distance_scaling_f32() {
        // 2400 m: rate 2.4f; 5 s base → (f32)50000 * 0.0001f = 5.0f; 2.4f * 5.0f.
        let rate = calc_ability_time_by_distance_rate(2400, 1000);
        assert_eq!(rate, 2.4f32);
        let base = 50000 as f32 * 0.0001f32;
        assert_eq!((rate * base).to_bits(), 12.0f32.to_bits());
        // 1800 m, 2.4 s base (raw 24000): (f32)24000 * 0.0001f = 2.3999999f; 1.8f * 2.3999999f = 4.3199997f (numpy f32).
        let t = calc_ability_time_by_distance_rate(1800, 1000) * (24000 as f32 * 0.0001);
        assert_eq!(t, 4.3199997f32);
        // cooldown 30 s on 1800 m: 54.0f exactly.
        assert_eq!(calc_cool_down_time_by_distance(300000 as f32 * 0.0001, 1800, 1000), 54.0f32);
        // infinite: -1 raw → -0.0001 * rate < 0
        assert!(rate * (-1i32 as f32 * 0.0001) < 0.0);
    }

    #[test]
    fn int_times_0001_is_not_decimal() {
        assert_eq!(4000 as f32 * 0.0001f32, 0.39999998f32);
    }

}
