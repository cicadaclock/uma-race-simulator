//! Test fixtures: bare `Race`s built without `race::init`.

use std::sync::OnceLock;

use crate::api::{HorseInput, RaceInput};
use crate::data::{DataEra, GameData};
use crate::rng::RaceRandom;
use crate::sim::ai::around::AroundTable;
use crate::sim::ai::order::Standings;
use crate::sim::horse::Horse;
use crate::sim::race::Race;
use crate::sim::skill::SkillBase;

/// The current era's full game data, loaded once.
pub fn data() -> &'static GameData {
    static D: OnceLock<GameData> = OnceLock::new();
    D.get_or_init(|| GameData::load(std::path::Path::new(crate::data::DATA_DIR), DataEra::current()).unwrap())
}

pub fn input(n: usize) -> RaceInput {
    RaceInput { horses: vec![HorseInput::default(); n], ..Default::default() }
}

fn bare_race<'a>(data: &'a GameData, input: &'a RaceInput, seed: i32, horses: Vec<Horse>) -> Race<'a> {
    Race {
        data,
        input,
        info: Default::default(),
        rng: RaceRandom::new(seed),
        horses,
        field: Default::default(),
        course: Default::default(),
        course_lane: Default::default(),
        event_manager: Default::default(),
        accumulate_time_since_start: 0.0,
        behind_lane_threshold: 0.0,
        world_per_metre: 0.0,
        distance_diff_max: 0.0,
        is_simulate_end: false,
        recorder: Default::default(),
        record_timer: 0.0,
        record_every_step: false,
        record_frames: true,
        next_modifier_id: 0,
    }
}

/// `n` default horses with only `race_paramdefine.json` loaded, course max lane 1.5, and the AI state's no-RNG
/// parts built (around horses, visible/near arrays).
pub fn ai_race(n: usize) -> Race<'static> {
    let s = std::fs::read_to_string(format!("{}/race_paramdefine.json", crate::data::DATA_DIR)).unwrap();
    let data: &'static GameData = Box::leak(Box::new(GameData { param: serde_json::from_str(&s).unwrap(), ..Default::default() }));
    let input: &'static RaceInput = Box::leak(Box::new(RaceInput::default()));
    let mut race = bare_race(data, input, 1, vec![Horse::default(); n]);
    race.info.lane_distance_max = 1.5;
    race.info.num_race_horses = n as i32;
    race.field.last_order = n as i32 - 1;
    race.field.standings = Standings::new(n);
    for i in 0..n {
        let h = &mut race.horses[i];
        h.lane_distance_max = 1.5;
        h.base_visible_distance = data.param.visibleDistance;
        h.ai.around = AroundTable::new(n, i);
        h.ai.visibility.visible_horses = vec![0; n - 1];
        h.ai.near_horses = vec![0; n - 1];
    }
    race
}

/// Full game data and one default horse per input horse, each with one skill holding two empty details.
pub fn skill_race(input: &RaceInput, seed: i32) -> Race<'_> {
    let mut horses = vec![Horse::default(); input.horses.len()];
    for (i, h) in horses.iter_mut().enumerate() {
        h.horse_data.horse_index = i as i32;
        h.final_corner_end_order = -1;
        h.skill_manager.skills.push(SkillBase::default());
    }
    bare_race(data(), input, seed, horses)
}
