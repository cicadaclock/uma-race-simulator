//! Horse AI: its state and creation. The around-horse scan is in [`around`]; the AI's per-frame order is in
//! [`crate::sim::tick`].

pub mod around;
pub mod compete_fight;
pub mod compete_top;
pub mod lane;
pub mod order;
pub mod overtake;
pub mod pacemaker;
pub mod position_keep;
pub mod temptation;

use crate::sim::ai::around::{AroundTable, Visibility};
use crate::sim::ai::lane::{ActiveLanePlanner, FixedLane, LanePlanner};
use crate::sim::ai::overtake::OvertakeTargets;
use crate::sim::ai::position_keep::PositionKeep;
use crate::sim::ai::temptation::Temptation;
use crate::sim::defs::last_spurt_calc_result;
use crate::sim::horse::Horse;
use crate::sim::physics::conserve_power::ConservePower;
use crate::sim::physics::speed::{BaseTargetSpeeds, TargetSpeed};
use crate::sim::race::Race;

/// Per-horse AI state. References to other horses (`block_horse`, `near_horses`, the overtake list, ...) are horse
/// indices.
#[derive(Clone, Debug, Default)]
pub struct Ai {
    pub target_speed: f32,
    pub target_lane: f32,
    pub after_goal_time: f32,
    pub around: Box<AroundTable>,
    pub block_horse: Option<usize>,
    pub block_horse_in: Option<usize>,
    pub block_horse_out: Option<usize>,
    pub is_around_horse_initialized: bool,
    pub in_lane_space: f32,
    pub out_lane_space: f32,
    pub overtake_cool_down_time: f32,
    pub is_down_slope_accel_mode: bool,
    pub down_slope_accel_check_sec: f32,
    pub position_keep: PositionKeep,
    pub temptation: Temptation,
    pub overtake_targets: OvertakeTargets,
    pub active_lane_planner: ActiveLanePlanner,
    /// Null when the course has no first move-lane point.
    pub lane_planner: LanePlanner,
    pub after_goal_lane: FixedLane,
    pub base_target_speeds: BaseTargetSpeeds,
    pub target_speed_calc: TargetSpeed,
    pub visibility: Visibility,
    pub conserve_power: ConservePower,
    pub is_force_speed_down_by_lane: bool,
    pub is_req_calc_last_spurt: bool,
    pub is_force_in_move_enable: bool,
    pub last_spurt_calc_count: i32,
    /// A `defs::last_spurt_calc_result` value.
    pub last_spurt_calc_result: i32,
    pub last_spurt_target_speed: f32,
    pub last_spurt_start_distance: f32,
    pub is_last_spurt_by_distance_requested: bool,
    pub is_surround_out: bool,
    pub is_surround_ahead: bool,
    pub is_surround_behind: bool,
    /// Horse indices; the first `near_horse_count` are valid.
    pub near_horses: Vec<usize>,
    pub near_horse_count: i32,
}

/// Creation order is load-bearing (the target-speed calculator draws RNG).
pub fn create_ai(race: &mut Race, i: usize) {
    race.horses[i].ai = Ai {
        last_spurt_calc_result: last_spurt_calc_result::FALSE,
        last_spurt_start_distance: -1.0,
        ..Default::default()
    };
    let n_around = race.horses.len() - 1;
    race.horses[i].ai.around = AroundTable::new(race.horses.len(), i);
    let c = PositionKeep::new(race, i);
    race.horses[i].ai.position_keep = c;
    let c = Temptation::new();
    race.horses[i].ai.temptation = c;
    let c = LanePlanner::new(race, i);
    race.horses[i].ai.lane_planner = c;
    let c = TargetSpeed::new(race, i); // RNG
    race.horses[i].ai.target_speed_calc = c;
    race.horses[i].ai.visibility = Visibility { visible_horses: vec![0; n_around], visible_horse_count: 0 };
    let c = BaseTargetSpeeds::new(race);
    race.horses[i].ai.base_target_speeds = c;
    let h = &mut race.horses[i];
    h.ai.target_speed = h.last_speed;
    h.ai.target_lane = h.lane_distance;
    h.ai.active_lane_planner = ActiveLanePlanner::Race;
    h.ai.near_horses = vec![0; n_around];
}

/// Time since the goal; switches `active_lane_planner` to OverRun after 1 s.
pub fn update_after_goal_time(race: &mut Race, i: usize, dt: f32) {
    let h = &mut race.horses[i];
    if h.is_finished {
        let old = h.ai.after_goal_time;
        h.ai.after_goal_time = dt + old;
        if 1.0 > old && h.ai.after_goal_time >= 1.0 {
            h.ai.active_lane_planner = ActiveLanePlanner::AfterGoal;
        }
    }
}

/// At least 1 s past the goal.
pub fn is_over_run_speed_down_mode(h: &Horse) -> bool {
    h.ai.after_goal_time >= 1.0
}
