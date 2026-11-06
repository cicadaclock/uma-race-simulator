//! Synthetic lobbies for differential tests (the GPU port against `sim::`) and fuzzing. Not game data: random
//! stats, styles, aptitudes and skills from the era's skill table, on race instances whose course files are present.

use crate::api::{HorseInput, RaceInput, SkillInput};
use crate::data::GameData;
use crate::data::course::lane_path_name;

/// A small deterministic generator (splitmix64), independent of the race's own RNG.
pub struct Rng(pub u64);

impl Rng {
    pub fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in lo..=hi.
    pub fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.u64() % (hi - lo + 1) as u64) as i32
    }
    pub fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[(self.u64() % v.len() as u64) as usize]
    }
}

/// Race instances (id, course set id) whose course path and event parameters are in `data`, sorted.
pub fn race_instances(data: &GameData) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    for ri in data.master.race_instance.values() {
        let Some(race) = data.master.race.get(&ri.race_id) else { continue };
        let Some(cs) = data.master.race_course_set.get(&race.course_set) else { continue };
        let path = lane_path_name(cs.race_track_id, cs.distance, cs.ground, cs.inout, 0);
        if data.course_paths.get(&path).is_some() && data.course_event_params.get(&cs.id.to_string()).is_some() {
            out.push((ri.id, cs.id));
        }
    }
    out.sort();
    out
}

/// One random lobby. `instances` from [`race_instances`]; `horses` = 0 picks 9..=18.
pub fn lobby(data: &GameData, instances: &[(i32, i32)], seed: u64, horses: usize) -> RaceInput {
    let mut r = Rng(seed);
    let &(race_instance_id, course_set_id) = r.pick(instances);
    let time = data.master.race_instance.get(&race_instance_id).map_or(2, |ri| ri.time.max(1));
    let mut skill_ids: Vec<i32> = data.skills.keys().copied().collect();
    skill_ids.sort();
    let n = if horses == 0 { r.range(9, 18) as usize } else { horses };
    let mut popularity: Vec<i32> = (1..=n as i32).collect();
    for i in (1..n).rev() {
        popularity.swap(i, (r.u64() % (i as u64 + 1)) as usize);
    }
    let horses = (0..n)
        .map(|i| {
            let style = r.range(1, 4);
            let mut proper_running_style = [0; 4];
            for (k, p) in proper_running_style.iter_mut().enumerate() {
                *p = if k as i32 + 1 == style { r.range(5, 8) } else { r.range(1, 8) };
            }
            let mut skills = Vec::new();
            for _ in 0..r.range(6, 30) {
                let id = *r.pick(&skill_ids);
                if !skills.iter().any(|s: &SkillInput| s.skill_id == id) {
                    skills.push(SkillInput { skill_id: id, level: r.range(1, 5) });
                }
            }
            HorseInput {
                chara_id: 1001 + r.range(0, 120),
                speed: r.range(600, 1800),
                stamina: r.range(400, 1600),
                pow: r.range(500, 1600),
                guts: r.range(300, 1400),
                wiz: r.range(300, 1500),
                running_style: style,
                motivation: r.range(1, 5),
                proper_distance: [r.range(3, 8), r.range(3, 8), r.range(3, 8), r.range(3, 8)],
                proper_running_style,
                proper_ground: [r.range(3, 8), r.range(3, 8)],
                skills,
                popularity: popularity[i],
                viewer_id: if i == 0 { 1 } else { 0 },
                frame_order: i as i32 + 1,
                ..Default::default()
            }
        })
        .collect();
    RaceInput {
        random_seed: r.u64() as i32,
        course_set_id,
        ground_condition: r.range(1, 4),
        weather: r.range(1, 4),
        season: r.range(1, 4),
        time,
        race_type: 8,
        horses,
        race_instance_id,
    }
}
