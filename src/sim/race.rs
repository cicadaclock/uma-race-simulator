//! Race state and initialisation. The frame loop is [`crate::sim::tick`].

use crate::sim::course::{self, CourseState, RaceEventManager};
use crate::sim::course::lanes::{self, CourseLane};
use crate::sim::field::{self, Field};
use crate::sim::record::Recorder;
use crate::sim::horse::Horse;
use crate::api::RaceInput;
use crate::data::GameData;
use crate::data::master::{RaceCourseSet, RaceTrack};
use crate::rng::RaceRandom;

/// Frame step in seconds. The game's float literal, not 1/15.
pub const DT: f32 = 0.0666;

/// Placeholder for a lookup `GameData` doesn't expose yet (e.g. item_data).
pub fn master<T>(what: &str) -> T {
    todo!("GameData lookup not wired yet: {what}")
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseDistances {
    pub phase_middle_start_distance: f32,
    pub phase_end_start_distance: f32,
    pub phase_last_start_distance: f32,
    pub course_distance: f32,
    pub is_initialized: bool,
}

impl PhaseDistances {
    /// Phases start at sections 4, 16 and 20 of 24.
    pub fn init(&mut self, course_distance: f32, section_distance: f32) {
        self.phase_middle_start_distance = section_distance * 4.0;
        self.phase_end_start_distance = section_distance * 16.0;
        self.phase_last_start_distance = section_distance * 20.0;
        self.course_distance = course_distance;
        self.is_initialized = true;
    }
}

#[derive(Clone, Debug, Default)]
pub struct RaceInfo {
    pub race_type: i32,
    pub random_seed: i32,
    pub score_calc_team_id: i32,
    pub race_course_set: RaceCourseSet,
    pub race_track: RaceTrack,
    pub initial_lane_type: i32,
    /// 1 (right-handed or straight-right) or 2.
    pub rotation_category: i32,
    /// Course distance / 24.
    pub course_section_distance: f32,
    /// 1 Short, 2 Mile, 3 Middle, 4 Long.
    pub course_distance_type: i32,
    pub course_furlong_num: i32,
    pub is_half_gate: bool,
    pub is_horse_num_variation_gate: bool,
    pub turf_vision_type: i32,
    pub ground_condition: i32,
    pub weather: i32,
    pub season: i32,
    pub time: i32,
    pub base_speed: f32,
    pub border_time_scaled: f32,
    pub num_race_horses: i32,
    pub lane_distance_max: f32,
    pub phase_distances: PhaseDistances,
    /// Popularity (0-based) → horse index.
    pub horse_index_by_popularity: Vec<i32>,
}

impl RaceInfo {
    pub fn new(input: &RaceInput, cs: RaceCourseSet, track: RaceTrack) -> RaceInfo {
        let mut ri = RaceInfo { base_speed: -1.0, race_type: input.race_type, random_seed: input.random_seed, ..Default::default() };
        ri.course_section_distance = cs.distance as f32 / 24.0;
        ri.initial_lane_type = track.initial_lane_type;
        ri.turf_vision_type = track.turf_vision_type;
        ri.rotation_category = if cs.turn == 1 || cs.turn == 3 { 1 } else { 2 };
        ri.course_furlong_num = crate::sim::util::f32_to_i32(cs.distance as f32 * 0.005);
        ri.course_distance_type = if cs.distance <= 1400 {
            1
        } else if cs.distance <= 1800 {
            2
        } else if cs.distance <= 2400 {
            3
        } else {
            4
        };
        ri.lane_distance_max = cs.float_lane_max as f32 * 0.0001;
        ri.ground_condition = input.ground_condition;
        ri.weather = input.weather;
        ri.season = input.season;
        ri.time = input.time;
        ri.phase_distances.init(cs.distance as f32, ri.course_section_distance);
        let enable_half_gate = track.enable_half_gate;
        let horse_num_gate_variation = track.horse_num_gate_variation;
        ri.race_course_set = cs;
        ri.race_track = track;
        let n = input.horses.len() as i32;
        ri.num_race_horses = n;
        ri.is_half_gate = n < 10 && enable_half_gate == 1;
        ri.is_horse_num_variation_gate = horse_num_gate_variation == 1;
        ri.horse_index_by_popularity = vec![0; input.horses.len()];
        for (i, h) in input.horses.iter().enumerate() {
            // Input popularity is 1-based; out-of-range values are skipped.
            if let Some(slot) = usize::try_from(h.popularity - 1).ok().and_then(|p| ri.horse_index_by_popularity.get_mut(p)) {
                *slot = i as i32;
            }
        }
        ri
    }
}

/// Borrowing idiom: `let data = race.data;` (a `&GameData` copy) before mutating `race`; read other horses
/// by copying values out of `race.horses[j]` first, then write `race.horses[i]`. A clone between `tick::step`s is an
/// independent race that continues bit for bit like the original (forking).
#[derive(Clone)]
pub struct Race<'a> {
    pub data: &'a GameData,
    pub input: &'a RaceInput,
    pub info: RaceInfo,
    /// Every draw goes through here.
    pub rng: RaceRandom,
    /// Indexed by horse index (= gate - 1).
    pub horses: Vec<Horse>,
    pub field: Field,
    pub course: CourseState,
    pub course_lane: CourseLane,
    pub event_manager: RaceEventManager,
    pub accumulate_time_since_start: f32,
    pub behind_lane_threshold: f32,
    /// World length of one course metre on lane 0.
    pub world_per_metre: f32,
    /// Largest distance spread between unfinished horses so far (recorded in the output).
    pub distance_diff_max: f32,
    pub is_simulate_end: bool,
    pub recorder: Recorder,
    /// Time left until the next frame record (`tick::step`).
    pub record_timer: f32,
    /// Not game behaviour (analysis hook): record a frame every step instead of the replay cadence
    /// (`record_interval`). Set it between `init` and `tick::start`; only the recorded frames change.
    pub record_every_step: bool,
    /// Not game behaviour (analysis hook): `false` records no frames, which nothing in the race reads. Set it between
    /// `init` and `tick::start`; only the recorded frames change.
    pub record_frames: bool,
    /// Source of `SkillParamModifier::id`, which stands in for object identity.
    pub next_modifier_id: u64,
}

pub fn lookup_course_set(data: &GameData, course_set_id: i32) -> RaceCourseSet {
    data.master.race_course_set.get(&course_set_id).unwrap_or_else(|| panic!("race_course_set {course_set_id} not in master")).clone()
}

pub fn lookup_race_track(data: &GameData, race_track_id: i32) -> RaceTrack {
    data.master.race_track.get(&race_track_id).unwrap_or_else(|| panic!("race_track {race_track_id} not in master")).clone()
}

/// In the game's init order. View, sound and commentary steps are omitted.
pub fn init<'a>(input: &'a RaceInput, data: &'a GameData) -> Race<'a> {
    let cs = lookup_course_set(data, input.course_set_id);
    let track = lookup_race_track(data, cs.race_track_id);
    let info = RaceInfo::new(input, cs, track);
    let seed = info.random_seed;
    let mut race = Race {
        data,
        input,
        info,
        // The game seeds it a few steps later; nothing draws before then.
        rng: RaceRandom::new(seed),
        horses: Vec::new(),
        field: Field::default(),
        course: CourseState::default(),
        course_lane: CourseLane::default(),
        event_manager: RaceEventManager::default(),
        accumulate_time_since_start: 0.0,
        behind_lane_threshold: 0.0,
        world_per_metre: 0.0,
        distance_diff_max: 0.0,
        is_simulate_end: false,
        recorder: Recorder::default(),
        record_timer: 0.0,
        record_every_step: false,
        record_frames: true,
        next_modifier_id: 0,
    };
    let params = course::load_course_params(data, &race.info);
    race.course = course::create_course_event(&params);
    init_race_base_speed(&mut race);
    let (lane_anim, over_run_anim) = lanes::load_course_lane_anims(data, &race.info);
    race.course_lane = CourseLane::new(race.info.rotation_category != 2, lane_anim, over_run_anim, race.info.race_course_set.distance);
    course::cache_straights(&mut race.course);
    race.course.corners = course::calc_corner_list(&race.course.event_param.course_event_corner, race.info.race_course_set.distance);
    race.course.slopes = course::calc_slope_list(&race.course.event_param.course_event_slope);
    field::init(&mut race);
    course::init_event_manager(&mut race);
    init_world_per_metre(&mut race);
    race.behind_lane_threshold = data.param.BehindPosCheckLaneDistanceMeterThreshold / 18.0;
    race
}

pub fn init_race_base_speed(race: &mut Race) {
    let p = &race.data.param;
    let dist = race.info.race_course_set.distance as f32;
    race.info.base_speed = p.raceBaseSpeedBaseSpeed - (dist - p.raceBaseSpeedBaseDistance) / p.raceBaseSpeedDistanceRate;
    let pc = &race.info.phase_distances;
    let end_len = pc.course_distance - pc.phase_end_start_distance;
    // One f64 expression, rounded at the store.
    let f = |x: f32| x as f64;
    race.info.border_time_scaled =
        (((f(dist) - f(end_len)) / f(race.info.base_speed) + f(end_len) / f(p.Speed.ExpectLastSpurtSpeed)) * f(p.Global.FinishTimeCoef)) as f32;
}

/// World distance between course positions 0 and 1 on lane 0.
pub fn init_world_per_metre(race: &mut Race) {
    let p0 = lanes::calc_world_transform(&race.course_lane, 0.0, 0.0).pos_world;
    let p1 = lanes::calc_world_transform(&race.course_lane, 1.0, 0.0).pos_world;
    let (dx, dy, dz) = (p0.x - p1.x, p0.y - p1.y, p0.z - p1.z);
    // The sum of squares in f64.
    race.world_per_metre = ((dy as f64 * dy as f64 + dx as f64 * dx as f64) + dz as f64 * dz as f64).sqrt() as f32;
}

pub fn update_distance_diff_max(race: &mut Race) {
    // With no unfinished horse the spread is -inf and never stored.
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for h in race.horses.iter().filter(|h| !h.is_finished) {
        if min > h.distance {
            min = h.distance;
        }
        if h.distance > max {
            max = h.distance;
        }
    }
    let d = max - min;
    if d > race.distance_diff_max {
        race.distance_diff_max = d;
    }
}

/// Seconds until the next recorded frame: every frame during the first second and while a player's horse
/// is within 25 m of the goal, otherwise once a second. Every frame with `Race::record_every_step`.
pub fn record_interval(race: &Race) -> f32 {
    if race.record_every_step {
        return 0.0;
    }
    let is_start_near = race.accumulate_time_since_start < 1.0;
    let goal = race.info.race_course_set.distance as f32;
    let is_goal_near = race.horses.iter().any(|h| h.horse_data.viewer_id != 0 && (goal - h.distance).abs() <= 25.0);
    if is_start_near || is_goal_near { 0.0666 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dt_bits() {
        assert_eq!(super::DT.to_bits(), 0x3D886595);
    }
}
