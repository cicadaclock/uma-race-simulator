//! Skill targets: which horses an ability applies to.
//!
//! Horses are referred to by index (= `HorseIndex`), so "not the owner" is an index compare.

use crate::sim::ai;
use crate::sim::defs::{RunningStyle, SkillModifierParam};
use crate::sim::field;
use crate::sim::race::Race;
use crate::sim::skill::SkillDetailRef;

/// Types 1..23 select horses; any other type selects nothing.
#[derive(Clone, Debug, Default)]
pub struct SkillTarget {
    /// 1 Self, 2 All, ... 23 ActivateHealSkill (see `select_targets`).
    pub target_type: i32,
    /// Horse index of the skill's owner.
    pub owner: usize,
    pub value_rh: i32,
    /// Horse indices.
    pub targets: Vec<usize>,
    pub is_debuff: bool,
}

impl SkillTarget {
    /// No RNG calls.
    pub fn new(owner: usize, target_type: i32, target_value: i32, is_debuff: bool) -> SkillTarget {
        SkillTarget { target_type, owner, value_rh: target_value, targets: Vec::new(), is_debuff }
    }
}

/// `ability` must have an effect. Some types draw RNG (RandomOtherSelf, *Temptation).
pub fn calc_targets(race: &mut Race, r: SkillDetailRef, ability: usize) {
    // The target object is taken out while computing: selectors only read horse/manager state, never an ability.
    let mut t = std::mem::take(target_slot(race, r, ability));
    calc_target(race, &mut t);
    *target_slot(race, r, ability) = t;
}

fn target_slot<'r>(race: &'r mut Race, r: SkillDetailRef, ability: usize) -> &'r mut SkillTarget {
    &mut race.horses[r.horse].skill_manager.skills[r.skill].details[r.detail].abilities[ability].as_mut().expect("ability without effect has no target").target
}

/// Debuffs drop horses with debuff cancel (the owner too) and the owner's teammates,
/// keeping the order.
pub fn calc_target(race: &mut Race, t: &mut SkillTarget) {
    if !(1..=23).contains(&t.target_type) {
        return;
    }
    select_targets(race, t);
    if t.is_debuff {
        let owner = t.owner;
        t.targets.retain(|&h| {
            let hh = &race.horses[h];
            if hh.skill_modifier_receiver.apply_modifier_bool(SkillModifierParam::DebuffCancel, false) {
                return false;
            }
            if h != owner {
                let a = race.horses[owner].horse_data.team_id;
                let b = hh.horse_data.team_id;
                if a != 0 && b != 0 && a == b {
                    return false;
                }
            }
            true
        });
    }
}

/// NaN sorts first (and equals NaN).
fn float_compare_to(l: f32, r: f32) -> i32 {
    if l < r {
        -1
    } else if l > r {
        1
    } else if l == r {
        0
    } else if l.is_nan() {
        if r.is_nan() { 0 } else { -1 }
    } else {
        1
    }
}

fn running_style_horses(race: &Race, v: i32) -> Vec<usize> {
    match RunningStyle::from_i32(v & 0xff) {
        RunningStyle::None => Vec::new(),
        s => field::running_style_horses(race, s).to_vec(),
    }
}

fn select_targets(race: &mut Race, t: &mut SkillTarget) {
    let owner = t.owner;
    let v = t.value_rh;
    let n_horses = race.horses.len();
    let cur_order = |race: &Race, h: usize| race.horses[h].cur_order;
    t.targets = match t.target_type {
        // Self
        1 => vec![owner],
        // All
        2 => (0..n_horses).collect(),
        // AllOtherSelf
        3 => (0..n_horses).filter(|&h| h != owner).collect(),
        // Visible: the first min(count, v) visible horses, stably sorted by 3D distance to the owner.
        4 => {
            let o = &race.horses[owner];
            let p = o.position;
            let vc = &o.ai.visibility;
            let n = if vc.visible_horse_count < v { vc.visible_horse_count } else { v };
            let mut list: Vec<(f32, usize)> = vc.visible_horses.iter().take(n.max(0) as usize)
                .map(|&h| {
                    let q = race.horses[h].position;
                    // This summation order is significant in f32.
                    let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z);
                    let s = (dy * dy + dx * dx) + dz * dz;
                    ((s as f64).sqrt() as f32, h)
                })
                .collect();
            list.sort_by(|a, b| float_compare_to(a.0, b.0).cmp(&0));
            list.into_iter().map(|(_, h)| h).collect()
        }
        // RandomOtherSelf: exactly cnt draws, each probing upward from the drawn index to the next free other horse.
        5 => {
            let n = race.field.horse_num;
            let cnt = if v >= n - 1 { n - 1 } else { v };
            let mut list: Vec<usize> = Vec::new();
            for _ in 0..cnt {
                let r = race.rng.random_int(n);
                for j in 0..n {
                    let idx = ((j + r) % n) as usize;
                    if idx != race.horses[owner].horse_data.horse_index as usize && !list.contains(&idx) {
                        list.push(idx);
                        break;
                    }
                }
            }
            list
        }
        // Order: the horse at rank v. Unused by master data.
        6 => (0..n_horses).find(|&h| cur_order(race, h) + 1 == v).into_iter().collect(),
        // OrderInfront / OrderBehind
        7 => (0..n_horses).filter(|&h| cur_order(race, h) + 1 < v).collect(),
        8 => (0..n_horses).filter(|&h| v < cur_order(race, h) + 1).collect(),
        // SelfInfront: the v ranks ahead of the owner, nearest first.
        9 => {
            let mut list = Vec::new();
            let mut o = cur_order(race, owner);
            if o != 0 {
                for _ in 0..v {
                    o -= 1;
                    if o < 0 {
                        break;
                    }
                    if let Some(h) = field::horse_at_order(race, o) {
                        list.push(h);
                    }
                }
            }
            list
        }
        // SelfBehind: the v ranks behind the owner, nearest first.
        10 => {
            let mut list = Vec::new();
            let mut o = cur_order(race, owner);
            if race.field.last_order != o {
                for _ in 0..v {
                    o += 1;
                    if race.field.last_order < o {
                        break;
                    }
                    if let Some(h) = field::horse_at_order(race, o) {
                        list.push(h);
                    }
                }
            }
            list
        }
        // TeamMember: the first team with the owner's id (none if absent); just the owner without a team.
        11 => {
            let tid = race.horses[owner].horse_data.team_id;
            if tid == 0 {
                vec![owner]
            } else {
                race.field.teams.iter().find(|ti| ti.team_id == tid).map_or_else(Vec::new, |ti| ti.members.clone())
            }
        }
        // Near: the v nearest by `lane gap * 18 + distance gap`, sorted with the game's (unstable) introsort.
        12 => {
            let ai = &race.horses[owner].ai;
            let cnt = ai.near_horse_count;
            let n = if v < cnt { v } else { cnt };
            let mut arr: Vec<usize> = ai.near_horses.iter().take(cnt.max(0) as usize).copied().collect();
            let key = |k: &usize| ai.around.lane_gap_abs.0[*k] * 18.0 + ai.around.distance_gap_abs.0[*k];
            crate::sim::util::intro_sort(&mut arr, |a, b| float_compare_to(key(a), key(b)));
            arr.into_iter().take(n.max(0) as usize).collect()
        }
        // SelfAndBlockFront (13), BlockSide (14)
        13 => std::iter::once(owner).chain(ai::around::block_horse_front(&race.horses[owner])).collect(),
        14 => {
            let ai = &race.horses[owner].ai;
            ai.block_horse_in.into_iter().chain(ai.block_horse_out).collect()
        }
        // NearInfront (15) / NearBehind (16): v unused, near-horse order, strict.
        15 | 16 => {
            let ai = &race.horses[owner].ai;
            let front = t.target_type == 15;
            ai.near_horses.iter().take(ai.near_horse_count.max(0) as usize)
                .copied()
                .filter(|&k| if front { ai.around.distance_gap.0[k] > 0.0 } else { 0.0 > ai.around.distance_gap.0[k] })
                .collect()
        }
        // RunningStyle (17), RunningStyleOtherSelf (18), RunningStyleTemptationOtherSelf (21: owner NOT excluded).
        17 => running_style_horses(race, v),
        18 => running_style_horses(race, v).into_iter().filter(|&h| h != owner).collect(),
        21 => running_style_horses(race, v).into_iter().filter(|&h| ai::temptation::is_temptation(&race.horses[h])).collect(),
        // SelfInfrontTemptation (19) / SelfBehindTemptation (20): n draws over the shrinking candidate list.
        19 | 20 => {
            let o = cur_order(race, owner);
            let infront = t.target_type == 19;
            if (infront && o == 0) || (!infront && race.field.last_order == o) {
                Vec::new()
            } else {
                let mut cand: Vec<usize> = race.field.rushed_horses.iter().copied()
                    .filter(|&h| if infront { cur_order(race, h) < o } else { o < cur_order(race, h) })
                    .collect();
                let n = if (cand.len() as i32) < v { cand.len() as i32 } else { v };
                let mut list = Vec::new();
                for _ in 0..n {
                    let r = race.rng.random_int(cand.len() as i32) as usize;
                    list.push(cand.remove(r));
                }
                list
            }
        }
        // CharaId (22)
        22 => (0..n_horses).filter(|&h| race.horses[h].horse_data.chara_id == v).collect(),
        // ActivateHealSkill (23): newest first, distinct, stop at exactly v.
        23 => {
            let mut list = Vec::new();
            for &h in race.field.heal_skill_horses.iter().rev() {
                if !list.contains(&h) {
                    list.push(h);
                    if list.len() as i32 == v {
                        break;
                    }
                }
            }
            list
        }
        _ => unreachable!(),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::test_util::ai_race;

    fn target(t: i32, v: i32, debuff: bool) -> SkillTarget {
        SkillTarget { target_type: t, owner: 0, value_rh: v, targets: Vec::new(), is_debuff: debuff }
    }

    #[test]
    fn near_sorted_by_score_and_capped() {
        let mut race = ai_race(4);
        let ai = &mut race.horses[0].ai;
        (ai.around.distance_gap_abs.0[1], ai.around.lane_gap_abs.0[1]) = (5.0, 0.0);
        (ai.around.distance_gap_abs.0[2], ai.around.lane_gap_abs.0[2]) = (1.0, 0.1); // 2.8
        (ai.around.distance_gap_abs.0[3], ai.around.lane_gap_abs.0[3]) = (0.5, 0.0);
        ai.near_horses = vec![1, 2, 3];
        ai.near_horse_count = 3;
        let mut t = target(12, 2, false);
        calc_target(&mut race, &mut t);
        assert_eq!(t.targets, vec![3, 2]);
    }

    #[test]
    fn random_other_self_probes_and_draws_cnt() {
        let mut race = ai_race(4);
        race.field.horse_num = 4;
        for (i, h) in race.horses.iter_mut().enumerate() {
            h.horse_data.horse_index = i as i32;
        }
        let mut t = target(5, 10, false);
        let before = race.rng.draws();
        calc_target(&mut race, &mut t);
        assert_eq!(race.rng.draws() - before, 3);
        let mut s = t.targets.clone();
        s.sort();
        assert_eq!(s, vec![1, 2, 3]);
    }

    #[test]
    fn debuff_filter_and_heal_history() {
        let mut race = ai_race(4);
        for (i, h) in race.horses.iter_mut().enumerate() {
            h.horse_data.team_id = [1, 1, 2, 0][i];
        }
        race.field.heal_skill_horses = vec![2, 1, 3, 1, 0];
        let mut t = target(23, 3, true);
        calc_target(&mut race, &mut t);
        // newest first, distinct, capped at 3 → [0, 1, 3]; debuff drops teammate 1, keeps owner.
        assert_eq!(t.targets, vec![0, 3]);
    }
}
