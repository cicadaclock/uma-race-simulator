//! Overtake targets: who a horse is trying to pass, the overtake cooldown, and the overtake-target timers.

use crate::sim::ai::around::block_horse_front;
use crate::sim::ai::lane;
use crate::sim::horse::Horse;
use crate::sim::race::Race;
use crate::sim::physics::speed::target_speed_with_skill;

/// The horses this horse is trying to overtake. The list persists between frames.
#[derive(Clone, Debug, Default)]
pub struct OvertakeTargets {
    /// Horse indices, in insertion order.
    pub targets: Vec<usize>,
}

/// Refreshes the overtake targets. While there are targets the cooldown is held full; once they are gone it runs
/// down, and on expiry forces one target-lane re-check. Runs after the goal too.
pub fn update_overtake(race: &mut Race, i: usize, dt: f32) {
    collect_overtake_target(race, i);
    let cool_down = race.data.param.overTakeCoolDownTime;
    let ai = &mut race.horses[i].ai;
    if ai.overtake_targets.targets.is_empty() {
        if ai.overtake_cool_down_time > 0.0 {
            ai.overtake_cool_down_time -= dt;
            if 0.0 >= ai.overtake_cool_down_time {
                lane::set_force_check_next_lane_one_time(race, i, true);
            }
        }
    } else {
        ai.overtake_cool_down_time = cool_down;
    }
}

/// Adds new visible targets, drops the ones that no longer qualify, and always keeps the front blocker.
fn collect_overtake_target(race: &mut Race, i: usize) {
    // `check_overtake_target` only reads, so the list can be taken out and put back.
    let mut list = std::mem::take(&mut race.horses[i].ai.overtake_targets.targets);
    let ai = &race.horses[i].ai;
    for &k in &ai.visibility.visible_horses[..ai.visibility.visible_horse_count as usize] {
        if check_overtake_target(race, i, k) && !list.contains(&k) {
            list.push(k);
        }
    }
    list.retain(|&k| check_overtake_target(race, i, k));
    if let Some(b) = ai.block_horse {
        if !list.contains(&b) {
            list.push(b);
        }
    }
    race.horses[i].ai.overtake_targets.targets = list;
}

/// Whether horse `j` is worth overtaking. Gaps come from the start-of-frame scan; speeds are live.
fn check_overtake_target(race: &Race, i: usize, j: usize) -> bool {
    let owner = &race.horses[i];
    let distance_gap = owner.ai.around.distance_gap.0[j];
    if distance_gap < -1.0 {
        return false;
    }
    let other = &race.horses[j];
    // The speed getters are pure, so they are evaluated only on the paths that use them.
    let my_target = || target_speed_with_skill(owner);
    let other_speed = other.last_speed;
    if block_horse_front(owner) == Some(j) {
        return other_speed < my_target();
    }
    let speed_gap = owner.ai.around.speed_gap.0[j];
    if distance_gap < 1.0 || distance_gap > 20.0 || speed_gap < 0.0 {
        return false;
    }
    if distance_gap / speed_gap > race.data.param.overTakeDistPerSpeed {
        return false;
    }
    if block_horse_front(other).is_none() {
        target_speed_with_skill(other) < my_target()
    } else {
        other_speed < my_target()
    }
}

/// How long a horse behind in order has had this horse as an overtake target.
pub fn update_overtake_target_time(race: &mut Race, i: usize, dt: f32) {
    let my_order = race.horses[i].cur_order;
    let targeted = race.horses.iter().any(|o| {
        my_order < o.cur_order && o.ai.overtake_targets.targets.contains(&i)
    });
    let h = &mut race.horses[i];
    if targeted {
        h.overtake_target_continue_time += dt;
    } else {
        h.overtake_target_continue_time = 0.0;
    }
}

/// How long this horse has had any overtake target.
pub fn update_overtake_target_have_time(race: &mut Race, i: usize, dt: f32) {
    let h = &mut race.horses[i];
    if h.ai.overtake_targets.targets.is_empty() {
        h.overtake_target_have_no_order_up_continue_time = 0.0;
        h.overtake_target_have_no_order_down_continue_time = 0.0;
    } else {
        h.overtake_target_have_no_order_up_continue_time += dt;
        h.overtake_target_have_no_order_down_continue_time += dt;
    }
}

/// Has an overtake target.
pub fn is_overtake(h: &Horse) -> bool {
    !h.ai.overtake_targets.targets.is_empty()
}

/// Has an overtake target, or the overtake cooldown is still running.
pub fn is_overtake_or_cool_down(h: &Horse) -> bool {
    is_overtake(h) || h.ai.overtake_cool_down_time > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::ai::around::{snapshot_scan_view, update_around_horses_param};
    use crate::sim::test_util::ai_race;
    use crate::sim::ai::lane::{RaceLanePlanner, LanePlanner};

    fn scan(race: &mut Race) {
        snapshot_scan_view(race);
        for i in 0..race.horses.len() {
            update_around_horses_param(race, i, 0.0666);
        }
    }

    fn force_check(race: &Race) -> bool {
        let LanePlanner::Race(c) = &race.horses[0].ai.lane_planner else { panic!() };
        c.is_force_check_next_lane_one_time
    }

    #[test]
    fn collect_and_cool_down() {
        let mut race = ai_race(3);
        for (i, (d, l, v)) in [(100.0, 0.3, 20.0), (105.0, 0.3, 18.0), (130.0, 0.3, 25.0)].into_iter().enumerate() {
            race.horses[i].distance = d;
            race.horses[i].lane_distance = l;
            race.horses[i].last_speed = v;
            race.horses[i].ai.target_speed = v;
        }
        race.horses[0].ai.lane_planner = LanePlanner::Race(RaceLanePlanner::default());
        scan(&mut race);
        // Horse 1: 5 m ahead, 2 m/s slower, 5/2 <= 15 → target. Horse 2: 30 m ahead (> 20 m, not visible) → not.
        update_overtake(&mut race, 0, 0.0666);
        let ai = &race.horses[0].ai;
        assert_eq!(ai.overtake_targets.targets, [1]);
        assert_eq!(ai.overtake_cool_down_time, race.data.param.overTakeCoolDownTime);
        update_overtake_target_time(&mut race, 1, 0.0666);
        assert_eq!(race.horses[1].overtake_target_continue_time, 0.0); // horse 0 is not behind horse 1 in order
        race.horses[0].cur_order = 1;
        update_overtake_target_time(&mut race, 1, 0.0666);
        assert_eq!(race.horses[1].overtake_target_continue_time, 0.0666);
        // Target pulls away faster: dropped from the list, cooldown runs down, then one forced lane check.
        race.horses[1].last_speed = 22.0;
        race.horses[1].ai.target_speed = 22.0;
        scan(&mut race);
        let mut frames = 0;
        while !force_check(&race) {
            update_overtake(&mut race, 0, 0.0666);
            assert!(race.horses[0].ai.overtake_targets.targets.is_empty());
            frames += 1;
        }
        assert_eq!(frames, 23); // 1.5 s / 0.0666 s
        assert!(race.horses[0].ai.overtake_cool_down_time <= 0.0);
    }

    #[test]
    fn front_blocker_always_listed() {
        // Front blocker 1 m ahead (gap < 1 m excluded by the speed test only when it is not the blocker).
        let mut race = ai_race(2);
        race.horses[0].distance = 100.0;
        race.horses[1].distance = 100.8;
        race.horses[0].lane_distance = 0.3;
        race.horses[1].lane_distance = 0.3;
        race.horses[0].ai.target_speed = 15.0;
        race.horses[1].last_speed = 20.0; // faster than our target: not a target, but it's the block horse
        scan(&mut race);
        assert_eq!(block_horse_front(&race.horses[0]), Some(1));
        collect_overtake_target(&mut race, 0);
        assert_eq!(race.horses[0].ai.overtake_targets.targets, [1]);
    }
}
