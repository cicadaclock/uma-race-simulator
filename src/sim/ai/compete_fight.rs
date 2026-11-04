//! Dueling (追い比べ): two horses side by side late in the race push each other on.

use crate::api::event_kind;
use crate::sim::race::Race;
use crate::sim::physics::stats;
use crate::sim::record;

/// Dueling update (after the overlap and spot struggle updates, before the pacemaker). Reads the candidate lists
/// built by `ai::around::update_around_horses_param` this frame.
///
/// Entry, per horse in index order, for a horse that is not dueling and has never dueled (one that crossed the goal
/// this frame still can): the first partner in its `compete_fight_near` (list order) with
/// - `compete_fight_candidate_continue_time >= TargetContinueTime`;
/// - the pre-update speed gap `speed_gap_abs < SpeedGap`;
/// - both HP >= `HpPer2` % of max HP;
/// - either one with `cur_order < N * TargetOrderPer / 100`, in float (last frame's 0-based order).
///
/// The partner must be dueling (join its group) or have never dueled (new group, both start). If the first partner
/// that passes the checks has already left a duel, the horse does not start this frame (no fallback to a later
/// entry). One dueling event per started horse, the starter first.
///
/// Exit, after the entries of the same frame: HP < `HpPer` %, or no other group member (including ones that already
/// left or finished) within `TargetContinueDistance`.
pub fn update(race: &mut Race) {
    let p = &race.data.param.CompeteFight;
    let n = race.horses.len();
    let top = n as f32 * p.TargetOrderPer as f32 * 0.01;
    let hp_ok = |h: &crate::sim::horse::Horse, per: i32| h.hp >= h.max_hp * per as f32 * 0.01;
    for i in 0..n {
        let h = &race.horses[i];
        if h.is_compete_fight || h.compete_fight_count > 0 {
            continue;
        }
        let partner = h.compete_fight_near.iter().map(|&k| h.ai.around.horse(k)).find(|a| {
            let o = &race.horses[a.info];
            a.compete_fight_candidate_continue_time >= p.TargetContinueTime
                && a.speed_gap_abs < p.SpeedGap
                && hp_ok(h, p.HpPer2)
                && hp_ok(o, p.HpPer2)
                && ((h.cur_order as f32) < top || (o.cur_order as f32) < top)
        });
        let Some(a) = partner else { continue };
        let o = a.info;
        if !race.horses[o].is_compete_fight && race.horses[o].compete_fight_count > 0 {
            continue; // the first qualifying partner has left its duel: no start
        }
        let g = if race.horses[o].is_compete_fight {
            race.horses[o].compete_fight_group
        } else {
            race.horses.iter().map(|h| h.compete_fight_group).max().unwrap_or(-1) + 1
        };
        start(race, i, g);
        if !race.horses[o].is_compete_fight {
            start(race, o, g);
        }
    }
    for i in 0..n {
        let h = &race.horses[i];
        if !h.is_compete_fight {
            continue;
        }
        let near = (0..n).any(|j| {
            j != i && race.horses[j].compete_fight_group == h.compete_fight_group
                && (race.horses[j].distance - h.distance).abs() <= p.TargetContinueDistance
        });
        if !near || h.hp < h.max_hp * p.HpPer as f32 * 0.01 {
            race.horses[i].is_compete_fight = false;
        }
    }
}

fn start(race: &mut Race, i: usize, group: i32) {
    let h = &mut race.horses[i];
    h.is_compete_fight = true;
    h.compete_fight_count += 1;
    h.compete_fight_group = group;
    let (idx, t) = (h.horse_data.horse_index as usize, race.recorder.cur_time);
    record::add_horse_event(race, event_kind::COMPETE_FIGHT, idx, t);
}

/// Target-speed add while dueling (from the frame after the start): `Pow(AddParam1Coef1 * Guts, AddParam1Coef2) *
/// AddParam1Coef3`, with the live Guts.
pub fn add_target_speed(race: &Race, i: usize) -> f32 {
    let h = &race.horses[i];
    if !h.is_compete_fight {
        return 0.0;
    }
    let p = &race.data.param.CompeteFight;
    crate::sim::util::mathf_pow(p.AddParam1Coef1 * stats::guts(h), p.AddParam1Coef2) * p.AddParam1Coef3
}

/// Accel add while dueling: `Pow(AddParam2Coef1 * Guts, AddParam2Coef2) * AddParam2Coef3`.
pub fn add_accel(race: &Race, i: usize) -> f32 {
    let h = &race.horses[i];
    if !h.is_compete_fight {
        return 0.0;
    }
    let p = &race.data.param.CompeteFight;
    crate::sim::util::mathf_pow(p.AddParam2Coef1 * stats::guts(h), p.AddParam2Coef2) * p.AddParam2Coef3
}
