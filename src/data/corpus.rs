//! horseACT saved race (.json in `Saved races/*/`) → `(RaceInput, Replay, date)`.
//! Both save formats are handled: new saves use camelCase keys, old ones the game's field names
//! (`<RandomSeed>k__BackingField`, `_raceInstanceMaster`, `Id`); the serde aliases below cover both.
//! Headless-client saves have no `time` (the race instance's is used), `horseIndex` or `runningStyleEx`.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::api::{HorseInput, RaceInput, SkillInput};
use crate::replay::{self, Replay};

/// Enum serialized by name (both formats use names; ints accepted too).
#[derive(Deserialize)]
#[serde(untagged)]
enum EnumVal {
    Int(i32),
    Name(String),
}

fn enum_val(x: &EnumVal, k: &str, names: &[(&str, i32)]) -> Result<i32, String> {
    match x {
        EnumVal::Int(n) => Ok(*n),
        EnumVal::Name(s) => names.iter().find(|(n, _)| n == s).map(|&(_, v)| v).ok_or_else(|| format!("{k}: unknown value {s:?}")),
    }
}

const GROUND_CONDITION: &[(&str, i32)] = &[("Good", 1), ("Soft", 2), ("Hard", 3), ("Bad", 4)];
const WEATHER: &[(&str, i32)] = &[("Sunny", 1), ("Cloudy", 2), ("Rainy", 3), ("Snow", 4)];
const SEASON: &[(&str, i32)] = &[("Spring", 1), ("Summer", 2), ("Fall", 3), ("Winter", 4), ("CherryBlossom", 5)];
const TIME: &[(&str, i32)] = &[("Morning", 1), ("Daytime", 2), ("Evening", 3), ("Night", 4)];
const RACE_TYPE: &[(&str, i32)] = &[
    ("None", 0), ("PvP", 1), ("Tutorial", 2), ("Story", 3), ("StoryCondition", 4), ("Champions", 5), ("Single", 6),
    ("SingleModeScenarioTeamRace", 7), ("RoomMatch", 8), ("Practice", 9), ("Daily", 10), ("TeamBuilding", 11),
    ("Legend", 12), ("ChallengeMatch", 13), ("TeamStadium", 14), ("Heroes", 16),
];

#[derive(Deserialize)]
struct SaveFile<'a> {
    #[serde(rename = "randomSeed", alias = "<RandomSeed>k__BackingField")]
    random_seed: i32,
    #[serde(rename = "raceCourseSet", alias = "<RaceCourseSet>k__BackingField")]
    race_course_set: IdOnly,
    #[serde(rename = "groundCondition", alias = "<GroundCondition>k__BackingField")]
    ground_condition: EnumVal,
    #[serde(rename = "weather", alias = "<Weather>k__BackingField")]
    weather: EnumVal,
    #[serde(rename = "season", alias = "<Season>k__BackingField")]
    season: EnumVal,
    #[serde(rename = "time", alias = "<Time>k__BackingField")]
    time: Option<EnumVal>,
    #[serde(rename = "raceType", alias = "<RaceType>k__BackingField")]
    race_type: EnumVal,
    #[serde(rename = "raceHorse", alias = "<RaceHorse>k__BackingField")]
    race_horse: Vec<SaveHorse>,
    #[serde(rename = "raceInstanceMaster", alias = "_raceInstanceMaster")]
    race_instance_master: RaceInstance,
    #[serde(borrow, rename = "simDataBase64", alias = "<SimDataBase64>k__BackingField")]
    sim_data_base64: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct IdOnly {
    #[serde(alias = "Id")]
    id: i32,
}

#[derive(Deserialize)]
struct RaceInstance {
    #[serde(alias = "Id")]
    id: i32,
    time: Option<i32>,
}

#[derive(Deserialize)]
struct SaveHorse {
    #[serde(rename = "horseIndex")]
    horse_index: Option<i32>,
    #[serde(rename = "responseHorseData", alias = "_responseHorseData")]
    response_horse_data: RaceHorseData,
}

/// `RaceHorseData` (snake_case in both formats).
#[derive(Deserialize)]
struct RaceHorseData {
    chara_id: i32,
    card_id: i32,
    speed: i32,
    stamina: i32,
    pow: i32,
    guts: i32,
    wiz: i32,
    running_style: i32,
    motivation: i32,
    proper_distance_short: i32,
    proper_distance_mile: i32,
    proper_distance_middle: i32,
    proper_distance_long: i32,
    proper_running_style_nige: i32,
    proper_running_style_senko: i32,
    proper_running_style_sashi: i32,
    proper_running_style_oikomi: i32,
    proper_ground_turf: i32,
    proper_ground_dirt: i32,
    skill_array: Vec<SkillInputJson>,
    item_id_array: Vec<i32>,
    team_id: i32,
    popularity: i32,
    npc_type: i32,
    final_grade: i32,
    trainer_name: Option<String>,
    rarity: i32,
    talent_level: i32,
    frame_order: i32,
    mob_id: i32,
    popularity_mark_rank_array: Vec<i32>,
    team_member_id: i32,
    team_rank: i32,
    single_mode_win_count: i32,
    #[serde(default)]
    fan_count: i32,
}

#[derive(Deserialize)]
struct SkillInputJson {
    skill_id: i32,
    level: i32,
}

fn horse(h: SaveHorse) -> HorseInput {
    let r = h.response_horse_data;
    HorseInput {
        chara_id: r.chara_id,
        card_id: r.card_id,
        speed: r.speed,
        stamina: r.stamina,
        pow: r.pow,
        guts: r.guts,
        wiz: r.wiz,
        running_style: r.running_style,
        motivation: r.motivation,
        proper_distance: [r.proper_distance_short, r.proper_distance_mile, r.proper_distance_middle, r.proper_distance_long],
        proper_running_style: [r.proper_running_style_nige, r.proper_running_style_senko, r.proper_running_style_sashi, r.proper_running_style_oikomi],
        proper_ground: [r.proper_ground_turf, r.proper_ground_dirt],
        skills: r.skill_array.iter().map(|s| SkillInput { skill_id: s.skill_id, level: s.level }).collect(),
        item_ids: r.item_id_array,
        team_id: r.team_id,
        popularity: r.popularity,
        npc_type: r.npc_type,
        final_grade: r.final_grade,
        viewer_id: r.trainer_name.is_some_and(|s| !s.is_empty()) as i64,
        rarity: r.rarity,
        talent_level: r.talent_level,
        frame_order: r.frame_order,
        mob_id: r.mob_id,
        popularity_mark_rank: r.popularity_mark_rank_array,
        team_member_id: r.team_member_id,
        team_rank: r.team_rank,
        single_mode_win_count: r.single_mode_win_count,
        fan_count: r.fan_count,
        muted_skills: Vec::new(),
        wiz_alt: None,
        muted_mechanics: 0,
        observed_skills: Vec::new(),
    }
}

pub fn parse_race(json: &str) -> Result<(RaceInput, Replay), String> {
    let v: SaveFile = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let b64 = v.sim_data_base64.as_deref().filter(|s| !s.is_empty()).ok_or("no replay (simDataBase64)")?;
    let replay = replay::decode_base64(b64)?;
    let mut horses = Vec::with_capacity(v.race_horse.len());
    for (i, h) in v.race_horse.into_iter().enumerate() {
        if h.horse_index.is_some_and(|x| x != i as i32) {
            return Err(format!("raceHorse[{i}] has horseIndex {:?}", h.horse_index));
        }
        horses.push(horse(h));
    }
    if replay.horse_num != horses.len() {
        return Err(format!("replay has {} horses, save has {}", replay.horse_num, horses.len()));
    }
    let input = RaceInput {
        random_seed: v.random_seed,
        course_set_id: v.race_course_set.id,
        ground_condition: enum_val(&v.ground_condition, "groundCondition", GROUND_CONDITION)?,
        weather: enum_val(&v.weather, "weather", WEATHER)?,
        season: enum_val(&v.season, "season", SEASON)?,
        time: match &v.time {
            Some(t) => enum_val(t, "time", TIME)?,
            None => v.race_instance_master.time.ok_or("no time")?,
        },
        race_type: enum_val(&v.race_type, "raceType", RACE_TYPE)?,
        horses,
        race_instance_id: v.race_instance_master.id,
    };
    Ok((input, replay))
}

/// `...-YYYYMMDD.json` → "YYYYMMDD".
pub fn date_from_path(path: &Path) -> Result<String, String> {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let d = stem.rsplit('-').next().unwrap_or("");
    if d.len() == 8 && d.bytes().all(|b| b.is_ascii_digit()) {
        Ok(d.to_string())
    } else {
        Err(format!("{}: no -YYYYMMDD suffix", path.display()))
    }
}

/// Every `.json` under `dir`, recursively, sorted.
pub fn race_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "json") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(dir, &mut files);
    files.sort();
    files
}

/// Loads a saved race: `(input, server replay, date "YYYYMMDD" from the filename)`.
pub fn load_race(path: &Path) -> Result<(RaceInput, Replay, String), String> {
    let date = date_from_path(path)?;
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (input, replay) = parse_race(&s).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((input, replay, date))
}
