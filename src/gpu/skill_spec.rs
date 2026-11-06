// The GPU skill program and skill state record formats (`gpu::skills`), shared with build.rs, which generates the
// GLSL constants (`skill_layout.glsl`) and the Rust ones (`skill_gen.rs`) from these lists. Every record is a run of
// u32 words; a name's index is its word offset.
//
// The program (binding 3) is static for a race: the parsed conditions, the abilities and their lookup tables, and
// where each piece of state lives. The state (binding 4) is one block of words per race: word w of the race in lane l
// is at `w * lanes + l`.

/// SkillTriggerKind, in its declaration order (`K_` + name).
pub const KINDS: &[&str] = &[
    "NULL",
    "COURSE_CORNER_COUNT",
    "GRADE",
    "RACE_ID",
    "TIME",
    "COURSE_DISTANCE",
    "IS_DIRT_GRADE",
    "RACE_ACCUMULATE_TIME",
    "SLOPE",
    "CORNER",
    "FURLONG",
    "ORDER",
    "ORDER_LAST_SPURT",
    "ORDER_FINAL_CORNER_END",
    "GATE_NUMBER",
    "POST_NUMBER",
    "HORSE_DISTANCE_RATE",
    "HORSE_REMAIN_DISTANCE",
    "HORSE_DISTANCE_DIFF_RATE",
    "OTHER_HORSE_BEHIND",
    "BE_BLOCKED",
    "BE_BLOCKED_FRONT",
    "BE_BLOCKED_SIDE",
    "BE_BLOCKED_ALL_CONTINUE_TIME",
    "BE_BLOCKED_FRONT_CONTINUE_TIME",
    "BE_BLOCKED_SIDE_CONTINUE_TIME",
    "BE_BLOCKED_FRONT_HP_EMPTY",
    "BE_BLOCKED_FRONT_HP_EMPTY_CONTINUE_TIME",
    "TEMPTATION_COUNT",
    "COMPETE_FIGHT_COUNT",
    "GOOD_START",
    "BAD_START",
    "BASE_SPEED",
    "BASE_STAMINA",
    "BASE_POWER",
    "BASE_GUTS",
    "BASE_WIZ",
    "HP_PER",
    "HP_EMPTY",
    "CURRENT_SPEED",
    "SPEED_ORDER",
    "VISIBLE_HORSE",
    "CHANGE_ORDER",
    "CHANGE_ORDER_ONE_TIME",
    "LANE_TYPE",
    "LANE_PER",
    "MOVE_LANE",
    "OVERTAKE",
    "OVERTAKE_TARGET_CONTINUE_TIME",
    "OVERTAKE_TARGET_NO_ORDER_UP_TIME",
    "OVERTAKE_TARGET_NO_ORDER_DOWN_TIME",
    "SURROUNDED",
    "NEAR_HORSE_COUNT",
    "NEAR_AROUND_ACCUMULATE_TIME",
    "DISTANCE_DIFF_TOP",
    "DISTANCE_DIFF_INFRONT",
    "DISTANCE_DIFF_BEHIND",
    "NEAR_FRONT_HORSE_COUNT",
    "NEAR_BEHIND_HORSE_COUNT",
    "HORSE_BEHIND_IN",
    "HORSE_BEHIND_OUT",
    "HORSE_INFRONT_NEAR_TIME",
    "HORSE_BEHIND_NEAR_TIME",
    "HORSE_INFRONT_NEAR_LANE_TIME",
    "HORSE_BEHIND_NEAR_LANE_TIME",
    "ACTIVATE_HEAL_SKILL_COUNT",
    "ACTIVATE_COUNT",
    "ACTIVATE_COUNT_LATER_HALF",
    "ACTIVATE_COUNT_ALL_TEAM",
    "CHANGE_ORDER_UP_PHASE_MIDDLE",
    "CHANGE_ORDER_UP_PHASE_END_AFTER",
    "CHANGE_ORDER_UP_CORNER",
    "CHANGE_ORDER_UP_LAST_SPURT",
    "CHANGE_ORDER_UP_FINAL_CORNER_AFTER",
    "CHANGE_ORDER_DOWN_PHASE_MIDDLE",
    "CHANGE_ORDER_DOWN_PHASE_END_AFTER",
    "CHANGE_ORDER_DOWN_CORNER",
    "CHANGE_ORDER_DOWN_LAST_SPURT",
    "CHANGE_ORDER_DOWN_FINAL_CORNER_AFTER",
    "GROUND_TYPE",
    "GROUND_CONDITION",
    "COURSE_ROTATION",
    "COURSE_DISTANCE_TYPE",
    "WEATHER",
    "SEASON",
    "RACE_TRACK_ID",
    "STRAIGHT_FRONT_TYPE",
    "FINAL_CORNER",
    "RUNNING_STYLE",
    "MOTIVATION",
    "POPULARITY",
    "PHASE",
    "TEMPTATION",
    "IS_LAST_SPURT",
    "LAST_SPURT",
    "ALWAYS",
    // p0: the value
    "LATCHED",
    // p0: whether the chara is in the race
    "EXIST_CHARA_ID",
    "HP",
    "BASHIN_DIFF_INFRONT",
    "BASHIN_DIFF_BEHIND",
    // p0: the float rh
    "DISTANCE_DIFF_TOP_FLOAT",
    // state: ok
    "RANDOM_LOT",
    // p0: lot_interval_max; state: is_lot_ok, lot_interval
    "RANDOM_LOT_OTHER_ACTIVATE",
    // state: the drawn second
    "RACE_ACCUMULATE_TIME_RANDOM",
    // p0: type 0..8
    "ORDER_RATE_IN_CONTINUE",
    "ORDER_RATE_OUT_CONTINUE",
    // p0: style, p1: only opponents
    "RUNNING_STYLE_TEMPTATION_COUNT",
    // p0: only opponents
    "TEMPTATION_BEHIND",
    "TEMPTATION_INFRONT",
    // state: prev_hp, is_initialized
    "HP_EMPTY_ONE_TIME",
    // p0: bit per horse with a viewer id
    "HORSE_VIEWER_ID_REMAIN_DISTANCE",
    // p0: the param set
    "HORSE_BEHIND_NEAR_LANE_TIME_PARAM_SET",
    // state: count, is_initialized
    "SAME_SKILL_HORSE_COUNT",
    // p0: skill id
    "USED_PARTICULAR_SKILL_ID",
    // p0: ability type
    "OTHER_CHARACTER_ACTIVATE_ADVANTAGE_SKILL",
    // state: enabled
    "ACTIVATE_OTHER_SKILL_DETAIL",
    // p0: bit per phase
    "ACTIVATE_COUNT_PHASE",
    // state: prev_activate_skill_count, activate_this_skill_count
    "IS_ACTIVATE_ANY_SKILL",
    // p0: program offset of the tags, p1: count
    "ACTIVATE_SKILL_TAG_GROUP",
    // state: prev
    "LAST_STRAIGHT_ONE_TIME",
    // p0: final corner exists, p1: min, p2: max (what the first update computes); state: is_initialized,
    // is_exist_final_corner, distance_min, distance_max
    "FINAL_CORNER_LATER_HALF",
    // p0: retries in update; state: distance_min, distance_max, is_initialized, has_distance
    "DISTANCE",
    // state: count, is_initialized, then 4 (min, max)
    "ALL_CORNER_RANDOM",
];

/// Record formats: (prefix, field names). `<PREFIX>_WORDS` is the record size.
pub const RECORDS: &[(&str, &[&str])] = &[
    // Program: the race, at word 0 of its block.
    (
        "PR",
        &[
            "horse_count",
            // 1: a horse has observed or muted skills (not supported on the GPU)
            "flags",
            "state_words",
            "corner_count",
            "grade",
            "race_id",
            "time",
            "is_dirt_grade",
            "ground",
            "ground_condition",
            "turn",
            "weather",
            "season",
            "track_id",
            "field_horse_num",
            // state offset of field.heal_skill_horses: count, then MAX_HORSES horses
            "heal_list",
            // program offset of the horse records (PH, one per horse)
            "horses",
            // program offsets of the tables of the random value usages 8, 9 (pairs coef, per) and counts
            "random1",
            "random1_count",
            "random2",
            "random2_count",
        ],
    ),
    // Program: a horse.
    (
        "PH",
        &[
            "skill_count",
            "skills",
            // state offsets of this horse's lists: count, then entries
            "used_ids",
            "prev_details",
            "activate_details",
            "reserved_ids",
            // field.prev_activated_skills entry of this horse (details of the last frame)
            "cached_details",
            "team_id",
            "chara_id",
            "post_number",
            "popularity",
            "final_grade",
            "base_speed",
            "base_stamina",
            "base_pow",
            "base_guts",
            "base_wiz",
            "base_wiz_skills",
            "motivation",
            // program offset / count of the team members (team order)
            "team",
            "team_count",
        ],
    ),
    // Program: a skill.
    (
        "PS",
        &[
            "id",
            "priority",
            "rarity",
            "is_general",
            // details[0].float_ability_time == -1
            "time_minus_one",
            // tags in AbilityValueUsageTagGroup1Array
            "is_specific",
            "tags",
            "tag_count",
            "detail0",
            "detail1",
            // state: cool_down_time, is_activate_enable
            "state",
        ],
    ),
    // Program: a skill detail.
    (
        "PD",
        &[
            "default_cool_down_time",
            "time_usage",
            "time_base",
            "time_rate",
            "is_time_infinite",
            "is_not_activate_skill",
            "is_use_detail_cool_down",
            // collections: program offset of the first term, term count, flags (1 Collection, 2 any_empty_group,
            // 4 has_lot_other)
            "pre",
            "pre_count",
            "pre_flags",
            "main",
            "main_count",
            "main_flags",
            "ability_count",
            "abilities",
            // the time usage's table (pairs threshold, value) and size; usage 2: (per distance, min), (max, 0);
            // usage 4: (max count, add time * distance rate)
            "time_table",
            "time_table_count",
            // state (AD)
            "state",
        ],
    ),
    // Program: a condition term.
    ("PT", &["kind", "op", "rh", "group", "state", "p0", "p1", "p2"]),
    // Program: an ability slot.
    (
        "PA",
        &[
            // 0: an ability without effect
            "present",
            "type",
            "usage",
            "base",
            // usages computed at construction: the final value
            "fixed_value",
            "is_fixed",
            // the usage's threshold table (pairs threshold, value) and size
            "table",
            "table_count",
            "target_type",
            "target_value",
            "target_is_debuff",
            "aa_type",
            // 0 none, 1 OrderUp, 2 ActivateAnySkill
            "aa_kind",
            "aa_max",
            // state (AA)
            "state",
        ],
    ),
    // State: a skill detail.
    ("AD", &["is_activated", "left_time", "cool_down_time", "need_pre", "increment_count", "additional_time"]),
    // State: an ability. Modifier handles are runs of consecutive modifier ids (one per activation).
    ("AA", &["value_on_activate", "additional_activate_count", "aa_count", "run_count", "runs"]),
];

/// Id runs an ability keeps (first, last id): one per activation until it stops.
pub const MAX_RUNS: u32 = 6;
