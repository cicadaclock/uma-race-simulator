//! What the stateless and latched triggers read from the race ([`value_lh`]).

use crate::sim::race::Race;
use crate::sim::skill::trigger::kind::SkillTriggerKind;
use crate::sim::util::{approximately, f32_to_i32, floor_to_int};
use crate::sim::ai::{around, overtake, temptation};
use crate::sim::field::{self, first_horse, horse_by_order};
use crate::sim::physics::last_spurt;

/// Lane band: 0 inside (<= 0.2), 1, 2, 3 outside (> 0.6).
fn lane_type(lane_distance: f32) -> i32 {
    if lane_distance <= 0.2 {
        0
    } else if 0.4 < lane_distance {
        (0.6 < lane_distance) as i32 + 2
    } else {
        1
    }
}

pub fn activate_skill_count(race: &Race, i: usize) -> i32 {
    race.horses[i].activate_skill_count_by_phase.iter().sum()
}

fn race_master<'a>(race: &Race<'a>) -> Option<&'a crate::data::master::Race> {
    let inst = race.data.master.race_instance.get(&race.input.race_instance_id)?;
    race.data.master.race.get(&inst.race_id)
}

/// Left-hand value of the stateless / latched triggers.
pub(super) fn value_lh(race: &Race, owner: usize, kind: &SkillTriggerKind) -> i32 {
    use SkillTriggerKind as K;
    let h = &race.horses[owner];
    let course_distance = race.info.race_course_set.distance;
    match kind {
        K::Latched(v) => *v,
        K::CourseCornerCount => race.course.corners.len() as i32,
        K::Grade => race_master(race).map_or(0, |m| m.grade),
        K::RaceId => race_master(race).map_or(0, |m| m.id),
        K::Time => race.info.time,
        K::CourseDistance => course_distance,
        K::IsDirtGrade => race_master(race).map_or(0, |m| m.is_dirtgrade),
        K::RaceAccumulateTime => floor_to_int(race.accumulate_time_since_start),
        K::Slope => h.slope_type as i32,
        K::Corner => h.cur_corner,
        K::Furlong => floor_to_int((h.distance + 0.5) * 0.005),
        K::Order => h.cur_order + 1,
        K::OrderLastSpurt => h.last_spurt_order + 1,
        K::OrderFinalCornerEnd => h.final_corner_end_order + 1,
        K::GateNumber => h.horse_data.horse_index + 1,
        K::PostNumber => h.horse_data.post_number,
        K::HorseDistanceRate => f32_to_i32(h.distance),
        K::HorseRemainDistance => course_distance - floor_to_int(h.distance),
        // How far behind the leader, in % of the leader-to-last span.
        K::HorseDistanceDiffRate => {
            let first = &race.horses[first_horse(race)];
            let last = &race.horses[field::last_horse_index(race) as usize];
            let span = first.distance - last.distance;
            let num = first.distance - h.distance;
            if approximately(span, 0.0) {
                0
            } else {
                floor_to_int(num / span * 100.0)
            }
        }
        K::OtherHorseBehind => race.info.num_race_horses - h.cur_order - 1,
        K::BeBlocked => around::is_block_all(h) as i32,
        K::BeBlockedFront => around::is_block_front(h) as i32,
        K::BeBlockedSide => around::is_block_side(h) as i32,
        K::BeBlockedAllContinueTime => {
            let f = h.block_front_continue_time;
            let s = h.block_side_continue_time;
            f32_to_i32(if s <= f { s } else { f })
        }
        K::BeBlockedFrontContinueTime => f32_to_i32(h.block_front_continue_time),
        K::BeBlockedSideContinueTime => f32_to_i32(h.block_side_continue_time),
        K::BeBlockedFrontHpEmpty => around::block_horse_front(h).is_some_and(|j| race.horses[j].hp <= 0.0) as i32,
        K::BeBlockedFrontHpEmptyContinueTime => f32_to_i32(h.block_front_hp_empty_continue_time),
        K::TemptationCount => h.ai.temptation.temptation_count,
        K::CompeteFightCount => h.compete_fight_count,
        K::GoodStart => h.is_good_start as i32,
        K::BadStart => h.is_bad_start as i32,
        K::BaseSpeed => f32_to_i32(h.horse_data.race_param.base_speed),
        K::BaseStamina => f32_to_i32(h.horse_data.race_param.base_stamina),
        K::BasePower => f32_to_i32(h.horse_data.race_param.base_pow),
        K::BaseGuts => f32_to_i32(h.horse_data.race_param.base_guts),
        K::BaseWiz => f32_to_i32(crate::sim::physics::stats::base_wiz_for(h, crate::api::wiz_mechanic::SKILLS)),
        K::HpPer => f32_to_i32(h.hp / h.max_hp * 100.0),
        K::HpEmpty => (h.hp <= 0.0) as i32,
        K::CurrentSpeed => f32_to_i32(h.last_speed),
        // Ranks by the Speed stat, not the current speed.
        K::SpeedOrder => {
            let own = crate::sim::physics::stats::speed(h);
            1 + race.horses.iter().filter(|o| crate::sim::physics::stats::speed(o) > own).count() as i32
        }
        K::VisibleHorse => h.ai.visibility.visible_horse_count,
        K::ChangeOrder | K::ChangeOrderOneTime => h.cur_order - h.prev_order,
        K::LaneType => lane_type(h.lane_distance),
        K::LanePer => floor_to_int(h.lane_distance_per_in_all_horses),
        // 0 until moving lanes for 1 s, then 1 inward, 2 outward.
        K::MoveLane => {
            if h.move_lane_continue_time < 1.0 {
                0
            } else if h.move_lane_continue_direction != crate::sim::defs::LaneDirection::In {
                2
            } else {
                1
            }
        }
        K::Overtake => overtake::is_overtake(h) as i32,
        K::OvertakeTargetContinueTime => floor_to_int(h.overtake_target_continue_time),
        K::OvertakeTargetHaveNoOrderUpContinueTime => floor_to_int(h.overtake_target_have_no_order_up_continue_time),
        K::OvertakeTargetHaveNoOrderDownContinueTime => floor_to_int(h.overtake_target_have_no_order_down_continue_time),
        K::Surrounded => around::is_surrounded(h) as i32,
        K::NearHorseCount => h.ai.near_horse_count,
        K::NearAroundAccumulateTime => floor_to_int(h.near_horse_around_accumulate_time),
        K::DistanceDiffTop => floor_to_int(race.horses[first_horse(race)].distance - h.distance),
        K::DistanceDiffInfront => floor_to_int(race.horses[horse_by_order(race, h.cur_order - 1)].distance - h.distance),
        K::DistanceDiffBehind => floor_to_int(h.distance - race.horses[horse_by_order(race, h.cur_order + 1)].distance),
        K::NearFrontHorseCount | K::NearBehindHorseCount => {
            let front = matches!(kind, K::NearFrontHorseCount);
            if h.ai.near_horse_count == 0 {
                return 0;
            }
            let mut c = 0;
            let mut i = 0;
            while (i as i32) < h.ai.near_horse_count {
                let g = h.ai.around.distance_gap.0[h.ai.near_horses[i]];
                if front {
                    if g > 0.0 {
                        c += 1;
                    }
                } else if !(g > 0.0 || g == 0.0) {
                    c += 1;
                }
                i += 1;
            }
            c
        }
        K::HorseBehindIn => (race.horses[horse_by_order(race, h.cur_order + 1)].lane_distance < h.lane_distance) as i32,
        K::HorseBehindOut => (h.lane_distance < race.horses[horse_by_order(race, h.cur_order + 1)].lane_distance) as i32,
        K::HorseInfrontNearTime => floor_to_int(h.in_front_horse_near_continue_time),
        K::HorseBehindNearTime => floor_to_int(h.behind_horse_near_continue_time),
        K::HorseInfrontNearLaneTime => floor_to_int(h.in_front_horse_near_lane_continue_time),
        K::HorseBehindNearLaneTime => floor_to_int(h.behind_horse_near_lane_continue_time),
        K::HorseBehindNearLaneTimeParamSet(p) => floor_to_int(h.behind_near_lane_times.get(*p).copied().unwrap_or(0.0)),
        K::ActivateHealSkillCount => h.activate_heal_skill_count,
        K::ActivateCount => activate_skill_count(race, owner),
        K::ActivateCountPhase(phases) => phases.iter().map(|&p| h.activate_skill_count_by_phase[p]).sum(),
        K::ActivateCountLaterHalf => h.activate_skill_count_by_distance[1],
        // Sum over the owner's team (0 without a team).
        K::ActivateCountAllTeam => {
            let t = h.horse_data.team_id;
            (0..race.horses.len())
                .filter(|&j| {
                    let tj = race.horses[j].horse_data.team_id;
                    tj != 0 && t != 0 && tj == t
                })
                .map(|j| activate_skill_count(race, j))
                .sum()
        }
        K::ChangeOrderUpPhaseMiddle => h.order_change_counter_up.change_count_phase_middle,
        K::ChangeOrderUpPhaseEndAfter => h.order_change_counter_up.change_count_phase_end_after,
        K::ChangeOrderUpCorner => h.order_change_counter_up.change_count_corner,
        K::ChangeOrderUpLastSpurt => h.order_change_counter_up.change_count_last_spurt,
        K::ChangeOrderUpFinalCornerAfter => h.order_change_counter_up.change_count_final_corner_after,
        K::ChangeOrderDownPhaseMiddle => h.order_change_counter_down.change_count_phase_middle,
        K::ChangeOrderDownPhaseEndAfter => h.order_change_counter_down.change_count_phase_end_after,
        K::ChangeOrderDownCorner => h.order_change_counter_down.change_count_corner,
        K::ChangeOrderDownLastSpurt => h.order_change_counter_down.change_count_last_spurt,
        K::ChangeOrderDownFinalCornerAfter => h.order_change_counter_down.change_count_final_corner_after,
        K::GroundType => race.info.race_course_set.ground,
        K::GroundCondition => race.info.ground_condition,
        K::CourseRotation => race.info.race_course_set.turn,
        K::CourseDistanceType => race.info.course_distance_type,
        K::Weather => race.info.weather,
        K::Season => race.info.season,
        K::RaceTrackId => race.info.race_course_set.race_track_id,
        K::StraightFrontType => h.cur_straight_front_type,
        K::FinalCorner => h.is_final_corner as i32,
        K::RunningStyle => h.horse_data.running_style as i32,
        K::Motivation => h.horse_data.race_param.motivation,
        // Stored 0-based.
        K::Popularity => h.horse_data.popularity + 1,
        K::Phase => h.phase as i32,
        K::Temptation => temptation::is_temptation(h) as i32,
        K::IsLastSpurt => last_spurt::is_last_spurt(h) as i32,
        K::LastSpurt => match h.ai.last_spurt_calc_result {
            1 => 2,
            2 => 1,
            _ => 0,
        },
        K::RunningStyleTemptationCount { style, is_check_only_opponent } => {
            let l = &race.field.rushed_horses_by_style[*style as usize];
            if *is_check_only_opponent {
                let t = h.horse_data.team_id;
                l.iter()
                    .filter(|&&j| {
                        let tj = race.horses[j].horse_data.team_id;
                        j != owner && (t == 0 || tj == 0 || t != tj)
                    })
                    .count() as i32
            } else {
                l.len() as i32
            }
        }
        K::TemptationBehind { is_check_only_opponent } | K::TemptationInfront { is_check_only_opponent } => {
            let behind = matches!(kind, K::TemptationBehind { .. });
            let t = h.horse_data.team_id;
            race.field
                .rushed_horses
                .iter()
                .filter(|&&j| {
                    let o = &race.horses[j];
                    let pass = !*is_check_only_opponent || t == 0 || o.horse_data.team_id == 0 || t != o.horse_data.team_id;
                    pass && if behind { o.cur_order > h.cur_order } else { o.cur_order < h.cur_order }
                })
                .count() as i32
        }
        K::HorseViewerIdRemainDistance(targets) => {
            let max = targets
                .iter()
                .map(|&j| race.horses[j].distance)
                .reduce(|a, b| if b > a { b } else { a })
                .expect("remain_distance_viewer_id: no horse with a viewer id (the game throws too)");
            course_distance - floor_to_int(max)
        }
        K::SameSkillHorseCount { same_skill_horse_count, .. } => *same_skill_horse_count,
        // The remaining kinds decide in `check_activate`.
        _ => 0,
    }
}
