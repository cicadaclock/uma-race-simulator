//! The field of horses: setup, overlap resolution, finish order and time clamp, end-of-frame caches and order
//! queries. The per-frame update order is in [`crate::sim::tick`].

use crate::sim::defs::RunningStyle;
use crate::sim::horse::{self, HorseData};
use crate::sim::ai::order::Standings;
use crate::sim::race::Race;
use crate::sim::ai::{compete_top::CompeteTop, pacemaker::Pacemaker};
use crate::sim::skill::SkillDetailRef;
use crate::sim::util::{approximately, f32_to_i32, intro_sort};
use crate::sim::ai::{self, lane, temptation};
use crate::sim::physics::{conserve_power, stats};
use crate::sim::skill;
use crate::rng::RaceRandom;

#[derive(Clone, Debug, Default)]
pub struct HorseTeamInfo {
    pub team_id: i32,
    /// Horse indices, sorted by team member id.
    pub members: Vec<usize>,
}

/// Per-running-style arrays are indexed by `RunningStyle as usize` (None=0 .. Oikomi=4).
#[derive(Clone, Debug, Default)]
pub struct Field {
    pub horse_num: i32,
    pub standings: Standings,
    /// Horses of each style, kept sorted by current order.
    pub same_running_style_horses: [Vec<usize>; 5],
    pub same_running_style_count: [i32; 5],
    pub running_style_top_order_horses: [Option<usize>; 5],
    pub running_style_last_order_horses: [Option<usize>; 5],
    pub rushed_horses_by_style: [Vec<usize>; 5],
    pub most_forward_running_style: RunningStyle,
    pub most_forward_style_horses: Vec<usize>,
    pub teams: Vec<HorseTeamInfo>,
    pub rushed_horses: Vec<usize>,
    /// Horses that activated skills last frame, in horse order, with those skill details.
    pub prev_activated_skills: Vec<(usize, Vec<SkillDetailRef>)>,
    /// Horses that activated a heal skill, most recent last.
    pub heal_skill_horses: Vec<usize>,
    pub pacemaker: Pacemaker,
    pub last_order: i32,
    pub is_finish_order_calculated: bool,
    pub is_defeat_checked: bool,
    pub horse_index_by_finish_order: Vec<usize>,
    pub compete_top: CompeteTop,
    /// Reused buffer for `update_overlap` (no per-frame allocation).
    pub sorted_by_lane_buf: Vec<usize>,
    /// Start-of-frame copy for the around-horse scans, see [`crate::sim::ai::around::snapshot_scan_view`].
    pub scan_view: crate::sim::ai::around::ScanView,
}

/// RNG draws happen in this order.
pub fn init(race: &mut Race) {
    let n = race.info.num_race_horses;
    race.field.horse_num = n;
    race.field.last_order = n - 1;
    race.field.pacemaker = Pacemaker::default();
    let input = race.input;
    for (idx, hi) in input.horses.iter().enumerate() {
        let data = HorseData::new(race, hi, idx);
        horse::construct(race, data);
    }
    cache_same_running_style_horses(race);
    cache_most_forward_running_style_horses(race);
    update_running_style_first_and_last_order_horses(race);
    // Skill triggers draw from the RNG, then the AI's target speed calculators, each in horse order.
    for i in 0..race.horses.len() {
        skill::create_skill(race, i);
    }
    for i in 0..race.horses.len() {
        ai::create_ai(race, i);
    }
    race.field.standings = Standings::new(race.horses.len());
    race.field.teams = create_team_info(race);
    race.field.horse_index_by_finish_order = vec![0; n as usize];
    race.field.compete_top = CompeteTop::default();
}

/// In horse order.
pub fn cache_same_running_style_horses(race: &mut Race) {
    for s in 1..5 {
        let horses: Vec<usize> = (0..race.horses.len()).filter(|&i| race.horses[i].horse_data.running_style as usize == s).collect();
        race.field.same_running_style_count[s] = horses.len() as i32;
        race.field.same_running_style_horses[s] = horses;
    }
}

pub fn cache_most_forward_running_style_horses(race: &mut Race) {
    for s in 1..5 {
        if let Some(&h) = race.field.same_running_style_horses[s].first() {
            race.field.most_forward_running_style = race.horses[h].horse_data.running_style;
            race.field.most_forward_style_horses = race.field.same_running_style_horses[s].clone();
            return;
        }
    }
}

/// Teams in first-appearance order, team 0 (no team) skipped; members sorted by team member id.
pub fn create_team_info(race: &Race) -> Vec<HorseTeamInfo> {
    let mut groups: Vec<HorseTeamInfo> = Vec::new();
    for (i, h) in race.horses.iter().enumerate() {
        let team_id = h.horse_data.team_id;
        match groups.iter_mut().find(|g| g.team_id == team_id) {
            Some(g) => g.members.push(i),
            None => groups.push(HorseTeamInfo { team_id, members: vec![i] }),
        }
    }
    groups.retain(|g| g.team_id != 0);
    for g in &mut groups {
        intro_sort(&mut g.members, |&a, &b| {
            race.horses[a].horse_data.team_member_id.wrapping_sub(race.horses[b].horse_data.team_member_id)
        });
    }
    groups
}

/// In the game's order (each RNG-drawing step noted).
pub fn init_before_start(race: &mut Race) {
    let n = race.horses.len();
    // Wiz lot RNG.
    for i in 0..n {
        skill::lot_activate_skill(race, i);
    }
    // First trigger group for all horses, then the second.
    for i in 0..n {
        skill::check_skill_trigger_and_activate_first_group(race, i);
    }
    for i in 0..n {
        skill::check_skill_trigger_and_activate_second_group(race, i);
    }
    // Items (special modes) are not applied: what they change (gate, field and mood conditions) is already in the
    // race data.
    for i in 0..n {
        conserve_power::update_activate_status(race, i);
    }
    // Start-delay RNG.
    let max = race.data.param.startDelayMax;
    for i in 0..n {
        horse::calc_delay_time(race, i, max);
    }
    // Base-target-speed RNG inside.
    recalc_params(race);
    // Rushed RNG.
    for i in 0..n {
        temptation::lot_temptation(race, i);
    }
    for i in 0..n {
        horse::update_world_transform_by_distance(race, i);
    }
}

pub fn recalc_params(race: &mut Race) {
    for i in 0..race.horses.len() {
        stats::recalc_params(race, i);
    }
    stats::calc_guts_order(race);
    stats::calc_wiz_order(race);
    for h in &mut race.horses {
        h.last_self_speed = h.last_speed;
    }
}

/// Pushes overlapping horses apart: walking outwards by lane, a horse within 0.4 m and one lane step of the
/// previous one moves to just outside it.
pub fn update_overlap(race: &mut Race) {
    // The comparator truncates lanes, so ties are common and the IntroSort permutation matters.
    let mut sorted = std::mem::take(&mut race.field.sorted_by_lane_buf);
    sorted.clear();
    sorted.extend(0..race.horses.len());
    intro_sort(&mut sorted, |&a, &b| {
        let (la, lb) = (f32_to_i32(race.horses[a].lane_distance), f32_to_i32(race.horses[b].lane_distance));
        la.wrapping_sub(lb).wrapping_mul(10000)
    });
    lane::set_force_check_next_lane_one_time(race, sorted[0], false);
    for k in 1..sorted.len() {
        let (prev, cur) = (sorted[k - 1], sorted[k]);
        let d = (race.horses[cur].distance - race.horses[prev].distance).abs();
        let l = (race.horses[cur].lane_distance - race.horses[prev].lane_distance).abs();
        // NaN counts as overlap.
        if !(d >= 0.4) && !(l >= 0.022222223) {
            race.horses[cur].lane_distance = race.horses[prev].lane_distance + 0.022222223;
            lane::set_force_check_next_lane_one_time(race, cur, true);
        }
    }
    race.field.sorted_by_lane_buf = sorted;
}

/// Finish order by finish time (ties and NaN by horse index), plus each horse's gap to the one ahead.
pub fn calc_finish_order(race: &mut Race) {
    if race.field.is_finish_order_calculated {
        return;
    }
    race.field.is_finish_order_calculated = true;
    let mut by_time: Vec<(usize, f64)> =
        race.horses.iter().enumerate().map(|(i, h)| (i, h.finish_time_scaled_for_finish_order_calc)).collect();
    intro_sort(&mut by_time, |x, y| match x.1.partial_cmp(&y.1) {
        Some(std::cmp::Ordering::Less) => -1,
        Some(std::cmp::Ordering::Greater) => 1,
        // Equal or NaN: NaN sorts first (double.CompareTo), then index ascending.
        _ => match (x.1.is_nan(), y.1.is_nan()) {
            (true, false) => -1,
            (false, true) => 1,
            _ => (x.0 as i32 - y.0 as i32).signum(),
        },
    });
    race.field.horse_index_by_finish_order = by_time.iter().map(|p| p.0).collect();
    for order in 0..by_time.len() {
        let i = by_time[order].0;
        race.horses[i].finish_order = order as i32;
        if order > 0 {
            let prev = race.horses[by_time[order - 1].0].finish_time_scaled;
            race.horses[i].finish_time_diff_from_prev_horse = race.horses[i].finish_time_scaled - prev;
        }
    }
}

/// Moves the winner's time into the course's allowed range (all horses shift by the same amount).
/// Draws from a fresh `RaceRandom(seed)`, not `race.rng`.
pub fn clamp_finish_time(race: &mut Race) {
    let order = race.field.horse_index_by_finish_order.clone();
    let first = order[0];
    let t0 = race.horses[first].finish_time_scaled;
    let Some(t) = calc_clamped_finish_time(t0, &race.info.race_course_set, race.info.random_seed) else { return };
    let diff = t - race.horses[first].finish_time_scaled;
    race.horses[first].finish_time_scaled = t;
    for &i in &order[1..] {
        race.horses[i].finish_time_scaled = race.horses[i].finish_time_scaled + diff;
    }
}

/// The clamped time (a random point near the violated bound), or None when `min <= t <= max`.
/// `cs` is the race instance's course set row. NaN clamps.
pub fn calc_clamped_finish_time(t: f32, cs: &crate::data::master::RaceCourseSet, seed: i32) -> Option<f32> {
    let min = cs.finish_time_min as f32 * 0.0001;
    let max = cs.finish_time_max as f32 * 0.0001;
    if t >= min && max >= t {
        return None;
    }
    let mut rng = RaceRandom::new(seed);
    Some(if min > t {
        let r = cs.finish_time_min_random_range as f32 * 0.0001;
        rng.random_float_range(-r, r) + min
    } else {
        let r = cs.finish_time_max_random_range as f32 * 0.0001;
        rng.random_float_range(-r, r) + max
    })
}

/// Accumulates the time the last-placed horse has spent in last place.
pub fn update_last_order_time(race: &mut Race, dt: f32) {
    let idx = last_horse_index(race);
    race.horses[idx as usize].last_order_time += dt;
}

/// Re-sorts each style's horses by current order (in place, persistent) and caches the first and last.
pub fn update_running_style_first_and_last_order_horses(race: &mut Race) {
    for s in 1..5 {
        let mut horses = std::mem::take(&mut race.field.same_running_style_horses[s]);
        intro_sort(&mut horses, |&a, &b| race.horses[a].cur_order.wrapping_sub(race.horses[b].cur_order));
        race.field.running_style_top_order_horses[s] = horses.first().copied();
        race.field.running_style_last_order_horses[s] = horses.last().copied();
        race.field.same_running_style_horses[s] = horses;
    }
}

/// Rushed horses per style and overall, style by style, each in current order.
pub fn cache_rushed_horses(race: &mut Race) {
    race.field.rushed_horses.clear();
    for s in 1..5 {
        race.field.rushed_horses_by_style[s].clear();
        for k in 0..race.field.same_running_style_horses[s].len() {
            let h = race.field.same_running_style_horses[s][k];
            if ai::temptation::is_temptation(&race.horses[h]) {
                race.field.rushed_horses_by_style[s].push(h);
                race.field.rushed_horses.push(h);
            }
        }
    }
}

/// Horses with no skill activated last frame are left out.
pub fn cache_prev_activated_skills(race: &mut Race) {
    race.field.prev_activated_skills.clear();
    for i in 0..race.horses.len() {
        let details = &race.horses[i].skill_manager.prev_activated_details;
        if !details.is_empty() {
            let details = details.clone();
            race.field.prev_activated_skills.push((i, details));
        }
    }
}

/// Moves every horse that just activated a heal skill to the end of the heal history.
pub fn cache_activate_heal_skill_horse_history_list(race: &mut Race) {
    for e in 0..race.field.prev_activated_skills.len() {
        let (h, details) = &race.field.prev_activated_skills[e];
        let h = *h;
        // A heal skill has an ability of type 9 with a positive value.
        let heal = details.iter().any(|&r| {
            skill::detail(race, r).abilities.iter().flatten().any(|ab| ab.ability_type == 9 && ab.ability_value_on_activate > 0.0)
        });
        if heal {
            let list = &mut race.field.heal_skill_horses;
            if let Some(p) = list.iter().position(|&x| x == h) {
                list.remove(p);
            }
            list.push(h);
        }
    }
}

/// Each unfinished horse's lane as a percentage of the field's lane spread (0 innermost, 100 outermost).
pub fn update_lane_distance_rate(race: &mut Race) {
    let mut min = f32::MAX;
    let mut max = -f32::MAX;
    for h in race.horses.iter().filter(|h| !h.is_finished) {
        let l = h.lane_distance;
        if min > l {
            min = l;
        }
        if l > max {
            max = l;
        }
    }
    let mut range = max - min;
    if approximately(0.0, range) {
        range = 1.0;
    }
    for h in race.horses.iter_mut().filter(|h| !h.is_finished) {
        // One f64 expression, rounded at the store.
        h.lane_distance_per_in_all_horses = ((h.lane_distance as f64 - min as f64) / range as f64 * 100.0) as f32;
    }
}

pub fn horse_at_order(race: &Race, order: i32) -> Option<usize> {
    usize::try_from(order).ok().and_then(|o| race.field.standings.horse_array_by_order.get(o)).map(|c| c.horse)
}

/// Horse at `order`; the callers' guards make a miss impossible (the game would crash on it too).
pub fn horse_by_order(race: &Race, order: i32) -> usize {
    horse_at_order(race, order).expect("no horse at this order")
}

/// Horse index at `order`, or 0 when out of range.
pub fn horse_index_at_order(race: &Race, order: i32) -> i32 {
    horse_at_order(race, order).map_or(0, |h| h as i32)
}

/// Horse at order 0 (index 0 when the order list is empty).
pub fn first_horse(race: &Race) -> usize {
    race.field.standings.horse_array_by_order.first().map_or(0, |c| c.horse)
}

pub fn last_horse_index(race: &Race) -> i32 {
    horse_index_at_order(race, race.horses.len() as i32 - 1)
}

pub fn running_style_horses<'r>(race: &'r Race, style: RunningStyle) -> &'r [usize] {
    &race.field.same_running_style_horses[style as usize]
}

pub fn running_style_count(race: &Race, style: RunningStyle) -> i32 {
    race.field.same_running_style_count[style as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_and_lane_rate() {
        let mut race = crate::sim::test_util::ai_race(3);
        for (i, (d, l)) in [(10.0, 0.5), (10.1, 0.51), (10.2, 0.52)].into_iter().enumerate() {
            race.horses[i].distance = d;
            race.horses[i].lane_distance = l;
        }
        update_overlap(&mut race);
        // all lanes truncate to 0 → index order kept; pushes cascade
        assert_eq!(race.horses[1].lane_distance, 0.5f32 + 0.022222223);
        assert_eq!(race.horses[2].lane_distance, (0.5f32 + 0.022222223) + 0.022222223);
        update_lane_distance_rate(&mut race);
        assert_eq!(race.horses[0].lane_distance_per_in_all_horses, 0.0);
        assert_eq!(race.horses[2].lane_distance_per_in_all_horses, 100.0);
    }

    #[test]
    fn finish_order_and_clamp() {
        let mut race = crate::sim::test_util::ai_race(3);
        for (i, t) in [95.5f32, 94.0, 95.5].into_iter().enumerate() {
            race.horses[i].finish_time_scaled = t;
            race.horses[i].finish_time_scaled_for_finish_order_calc = t as f64;
        }
        calc_finish_order(&mut race);
        assert_eq!(race.field.horse_index_by_finish_order, [1, 0, 2]);
        assert_eq!(race.horses[0].finish_time_diff_from_prev_horse, 1.5);
        assert_eq!(race.horses[2].finish_time_diff_from_prev_horse, 0.0);
        let mut cs = crate::data::master::RaceCourseSet::default();
        (cs.finish_time_min, cs.finish_time_max, cs.finish_time_min_random_range) = (950000, 1000000, 5000);
        assert_eq!(calc_clamped_finish_time(96.0, &cs, 1), None);
        let t = calc_clamped_finish_time(94.0, &cs, 1).unwrap();
        assert!((94.5..=95.5).contains(&t));
        race.info.race_course_set = cs;
        race.info.random_seed = 1;
        clamp_finish_time(&mut race);
        assert_eq!(race.horses[1].finish_time_scaled, t);
        assert_eq!(race.horses[0].finish_time_scaled, 95.5 + (t - 94.0));
    }
}
