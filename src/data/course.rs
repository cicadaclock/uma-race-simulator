//! Course event params and course lane paths, as parsed data. Geometry logic lives in the sim.
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Deserialize;

/// One course event. `_paramType`: 0 Corner, 1 GroundChange, 2 Straight, 3 LaneMaxChange, ..., 7 MoveLanePoint, 11 Slope.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CourseParam {
    pub _paramType: i32,
    pub _values: Vec<i32>,
    pub _distance: f32,
}

/// A course's events (data/courseeventparam/<race_course_set.id>.json), in file order (unsorted).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CourseEvents {
    pub courseParams: Vec<CourseParam>,
}

/// Rotation math is in `sim::course::lanes`.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quaternion {
    fn default() -> Self {
        Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }
}

/// Lane path keyframes. `valueY` may be empty (= 0).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CourseLaneKey {
    pub valueX: Vec<f32>,
    pub valueY: Vec<f32>,
    pub valueZ: Vec<f32>,
    pub rotation: Vec<Quaternion>,
}

/// A lane path (data/coursepath/an_pos_race....json).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CourseLaneAnim {
    pub length: f32,
    pub Distance: i32,
    pub key: CourseLaneKey,
}

/// File stem in data/coursepath/ of a course's lane path; `path_type` 0 = normal race.
pub fn lane_path_name(race_track_id: i32, distance: i32, ground: i32, inout: i32, path_type: i32) -> String {
    format!("an_pos_race{race_track_id:04}_00_{distance:04}_{:02}_{}_{path_type}", ground - 1, inout - 1)
}

/// File stem of the track's overrun (round) path, the `_01_0000_` variant. Its inout digit is always 0, whatever the
/// course (with `inout - 1`, inner/outer courses found no overrun path).
pub fn round_lane_path_name(race_track_id: i32, ground: i32, path_type: i32) -> String {
    format!("an_pos_race{race_track_id:04}_01_0000_{:02}_0_{path_type}", ground - 1)
}

fn read_json<T: for<'de> Deserialize<'de>>(p: &Path) -> Result<T, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

/// `<stem>.json` files of a list of directories, parsed on first `get` and cached (a race needs 1 event-param file
/// and 2 lane paths out of ~26 MB; parsing them all took ~90 ms per GameData). Later directories take precedence.
#[derive(Debug, Default)]
pub struct JsonFiles<T> {
    dirs: Vec<PathBuf>,
    cache: Mutex<HashMap<String, Option<Arc<T>>>>,
}

impl<T: for<'de> Deserialize<'de>> JsonFiles<T> {
    pub fn new(dirs: Vec<PathBuf>) -> Result<JsonFiles<T>, String> {
        if let Some(d) = dirs.iter().find(|d| !d.is_dir()) {
            return Err(format!("{}: not a directory", d.display()));
        }
        Ok(JsonFiles { dirs, cache: Mutex::default() })
    }

    /// `None` if no directory has `<stem>.json`. Panics on an unreadable / invalid file.
    pub fn get(&self, stem: &str) -> Option<Arc<T>> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = cache.get(stem) {
            return v.clone();
        }
        // The cache outlives any arena scope of the caller.
        crate::arena::outside(|| {
            let path = self.dirs.iter().rev().map(|d| d.join(format!("{stem}.json"))).find(|p| p.is_file());
            let v = path.map(|p| Arc::new(read_json(&p).unwrap_or_else(|e| panic!("{e}"))));
            cache.insert(stem.to_string(), v.clone());
            v
        })
    }
}
