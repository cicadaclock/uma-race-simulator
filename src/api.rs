//! Public input/output types.

#[derive(Clone, Debug, Default)]
pub struct RaceInput {
    pub random_seed: i32,
    /// `race_course_set.id` (master.mdb).
    pub course_set_id: i32,
    /// Ground condition: 1 Good, 2 Soft (やや重), 3 Hard (重), 4 Bad (不良).
    pub ground_condition: i32,
    /// 1 Sunny, 2 Cloudy, 3 Rainy, 4 Snow.
    pub weather: i32,
    /// 1 Spring, 2 Summer, 3 Fall, 4 Winter, 5 CherryBlossom.
    pub season: i32,
    /// Time of day: 1 Morning, 2 Daytime, 3 Evening, 4 Night (skill conditions only).
    pub time: i32,
    /// Race type (skill conditions, career stat bonus): 1 PvP, 2 Tutorial, 3 Story, 4 StoryCondition, 5 Champions,
    /// 6 Single (career), 7 SingleModeScenarioTeamRace, 8 RoomMatch, 9 Practice, 10 Daily, 11 TeamBuilding, 12 Legend,
    /// 13 ChallengeMatch, 14 TeamStadium, 16 Heroes.
    pub race_type: i32,
    /// Horses in gate order (index i = gate i+1 = horse index).
    pub horses: Vec<HorseInput>,
    /// `race_instance.id` (master.mdb), which gives the `race` row (grade / race id skill conditions).
    pub race_instance_id: i32,
}

#[derive(Clone, Debug, Default)]
pub struct HorseInput {
    pub chara_id: i32,
    pub card_id: i32,
    /// Raw stats (before the >1200 halving and motivation).
    pub speed: i32,
    pub stamina: i32,
    pub pow: i32,
    pub guts: i32,
    pub wiz: i32,
    /// 1 Nige, 2 Senko, 3 Sashi, 4 Oikomi.
    pub running_style: i32,
    /// 1 (worst) ..= 5 (best).
    pub motivation: i32,
    /// Aptitude grades 1 (G) ..= 8 (S): [short, mile, middle, long].
    pub proper_distance: [i32; 4],
    /// [nige, senko, sashi, oikomi].
    pub proper_running_style: [i32; 4],
    /// [turf, dirt].
    pub proper_ground: [i32; 2],
    /// Skills in processing order (the order matters).
    pub skills: Vec<SkillInput>,
    pub item_ids: Vec<i32>,
    pub team_id: i32,
    pub popularity: i32,
    pub npc_type: i32,
    pub final_grade: i32,
    /// Non-zero for player horses (incl. ghosts), 0 otherwise. Only `!= 0` matters (the dense recording window near
    /// the goal).
    pub viewer_id: i64,
    pub rarity: i32,
    pub talent_level: i32,
    pub frame_order: i32,
    pub mob_id: i32,
    /// `popularity_mark_rank_array` (newspaper marks Left/Center/Right; 0 = top mark).
    pub popularity_mark_rank: Vec<i32>,
    pub team_member_id: i32,
    /// Team rank (team stat bonus).
    pub team_rank: i32,
    pub single_mode_win_count: i32,
    /// 0 when absent.
    pub fan_count: i32,
    /// Not game data (analysis hook): ids of skills in `skills` whose abilities apply no effect. Everything else
    /// runs as normal (wiz lot, triggers, targets, ability values, all RNG draws, events, durations, cooldowns,
    /// activation counts that other conditions read), so a "skill" vs "skill muted" pair shares one RNG sequence
    /// up to the first effect. Empty = normal behaviour. Gate and edge cases: `sim::skill::ability::activate`.
    pub muted_skills: Vec<i32>,
    /// Not game data (analysis hook): a second raw Wit that only the mechanics in its mask read, through the same
    /// halving, motivation, style aptitude and skill modifiers. With `wiz` at one value and this at another, a pair of
    /// races differs in one mechanic's Wit only. None = normal behaviour.
    pub wiz_alt: Option<WizAlt>,
    /// Not game data (analysis hook): bits of [`mechanic`] that run inert for this horse. 0 = normal behaviour.
    pub muted_mechanics: u8,
    /// Not game data (analysis hook): ids of skills in `skills` that draw from their own generators, seeded from the
    /// race seed and the skill id, and stay out of the random activation draw, so that holding them moves no other
    /// draw. One that is also in `muted_skills` is silent: everything else in the race treats it as absent, so the race
    /// runs bit for bit as without it, while its own triggers and activations still run. A skill that is not silent
    /// activates as it does when silent, up to its first effect. Empty = normal behaviour.
    pub observed_skills: Vec<i32>,
}

impl HorseInput {
    /// Whether skill `id` is silent, see [`HorseInput::observed_skills`].
    pub fn is_silent_skill(&self, id: i32) -> bool {
        self.observed_skills.contains(&id) && self.muted_skills.contains(&id)
    }
}

/// See [`HorseInput::wiz_alt`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WizAlt {
    /// Raw Wit, like [`HorseInput::wiz`].
    pub wiz: i32,
    /// Bits of [`wiz_mechanic`].
    pub mechanics: u8,
}

/// Everything the sim reads Wit for (the live Wit order is only reported, so it keeps the real Wit).
pub mod wiz_mechanic {
    /// The per-section base target speed rolls.
    pub const SECTIONS: u8 = 1 << 0;
    /// Position keep's Wit rolls (speed up, overtake, pace up).
    pub const POSITION_KEEP: u8 = 1 << 1;
    /// Picking a last spurt candidate.
    pub const LAST_SPURT: u8 = 1 << 2;
    /// Skill Wit lots (pre-race and random_lot_other_activate), base_wiz conditions and Wit-scaled values.
    pub const SKILLS: u8 = 1 << 3;
    /// The rush (掛かり) lottery.
    pub const RUSHED: u8 = 1 << 4;
    /// Downhill accel mode.
    pub const DOWNHILL: u8 = 1 << 5;
    /// The course-set speed bonus for Wit over its thresholds.
    pub const COURSE_BONUS: u8 = 1 << 6;
    pub const COUNT: usize = 7;
    pub const ALL: u8 = (1 << COUNT) - 1;
}

/// Mechanics [`HorseInput::muted_mechanics`] can switch off.
pub mod mechanic {
    /// Conserve power: nothing stored, nothing released (it draws no RNG, so a muted pair matches up to the release).
    pub const CONSERVE_POWER: u8 = 1 << 0;
    pub const COUNT: usize = 1;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkillInput {
    pub skill_id: i32,
    pub level: i32,
}

/// Simulation output: frames, results and events, un-quantized. `replay::quantize` rounds it to recorded precision.
/// `spans` and `skill_spans` are sim-only (not in replays).
#[derive(Clone, Debug, Default)]
pub struct SimOutput {
    pub frames: Vec<Frame>,
    pub results: Vec<HorseResult>,
    pub events: Vec<Event>,
    pub spans: Vec<Span>,
    pub skill_spans: Vec<SkillSpan>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    pub time: f32,
    pub horses: Vec<HorseFrame>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HorseFrame {
    pub distance: f32,
    pub lane_position: f32,
    pub speed: f32,
    pub hp: f32,
    pub temptation_mode: i8,
    pub block_front_horse_index: i8,
    /// Sim-only (not in replays, `replay::quantize_frame` leaves them out): the skill effects on the horse now, summed
    /// over every active modifier, its own skills' and others' debuffs alike. Target speed + current speed (m/s).
    pub skill_speed: f32,
    /// Acceleration (m/s²).
    pub skill_accel: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HorseResult {
    pub finish_order: i32,
    pub finish_time_raw: f32,
    pub finish_time_scaled: f32,
    pub finish_diff_time: f32,
    pub start_delay_time: f32,
    pub last_spurt_start_distance: f32,
    /// Serialized as byte.
    pub guts_order: i32,
    /// Serialized as byte.
    pub wiz_order: i32,
    /// Serialized as byte.
    pub running_style: i32,
    /// Defeat reason (1 Win, 2 Lose, 5 GutsOrder, 9 PassiveSkillNum, 13 ProperGround, ...).
    pub defeat: i32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Event {
    pub time: f32,
    /// One of [`event_kind`].
    pub kind: u8,
    pub params: Vec<i32>,
}

/// A stretch of one horse in one state, sampled every step: from the first step in it to the first step out of it (or
/// the horse's finish). Sim-only: replays carry start events at most.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Span {
    pub horse: usize,
    /// One of [`span_kind`].
    pub kind: u8,
    pub start: f32,
    pub end: f32,
}

/// One skill effect of one horse, from its activation to when the effect ended: its duration as the race ran it
/// (scaled by distance, extended while its extension conditions held, or stopped early). One per activation of a
/// skill detail; skills that activate without an effect have none. Sim-only.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SkillSpan {
    /// Gate index, as in skill events.
    pub horse: usize,
    pub skill_id: i32,
    /// Skill detail (0 or 1), as in skill events.
    pub detail: i32,
    pub start: f32,
    /// None = still running when the race ended: `infinite`, or past the horse's finish.
    pub end: Option<f32>,
    /// No time limit (passive skills): it lasts the whole race.
    pub infinite: bool,
}

pub mod span_kind {
    /// Dueling (追い比べ).
    pub const DUEL: u8 = 0;
    /// Spot struggle (位置取り争い).
    pub const SPOT_STRUGGLE: u8 = 1;
    /// Rushed (掛かり).
    pub const RUSHED: u8 = 2;
    /// A horse ahead in its lane caps its speed.
    pub const BLOCKED: u8 = 3;
    /// The released conserve-power accel boost.
    pub const CONSERVE_POWER: u8 = 4;
    /// Downhill accel mode.
    pub const DOWNHILL: u8 = 5;
    /// Position keep modes.
    pub const SPEED_UP: u8 = 6;
    pub const OVERTAKE: u8 = 7;
    pub const PACE_UP: u8 = 8;
    pub const PACE_DOWN: u8 = 9;
    pub const PACE_UP_EX: u8 = 10;
    pub const COUNT: usize = 11;
}

pub mod event_kind {
    pub const SKILL: u8 = 3;
    pub const COMPETE_TOP: u8 = 4;
    pub const COMPETE_FIGHT: u8 = 5;
    pub const CONSERVE_POWER_RELEASE: u8 = 6;
}
