//! The GPU race state layout (generated from `layout_spec.rs`) and one lane's state on the CPU side.

#[allow(dead_code)]
mod spec {
    include!("layout_spec.rs");
}
pub use spec::{
    BTS_SECTIONS, FIELDS, Kind, MAX_BEHIND_NEAR_PARAMS, MAX_HORSES, MAX_MODIFIERS, MAX_ONE_TIME_MODIFIERS, ORDER_IN_OUT_TYPES,
    PK_STRATEGIES, RNG_WORDS, Scope,
};

/// A field of the state: its first plane and shape. Constants `H_<NAME>` / `R_<NAME>` are generated.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub index: usize,
    pub base: u32,
    pub kind: Kind,
    pub scope: Scope,
    pub len: u32,
    pub name: &'static str,
}

impl Field {
    pub const fn words(self) -> u32 {
        if matches!(self.kind, Kind::F64) { 2 } else { 1 }
    }
    /// Plane of horse `h`'s value.
    pub const fn horse(self, h: usize) -> u32 {
        self.base + h as u32 * self.words()
    }
    /// Plane of element `k` of horse `h`'s array.
    pub const fn horse_k(self, h: usize, k: usize) -> u32 {
        self.base + (k as u32 * MAX_HORSES + h as u32) * self.words()
    }
    pub const fn race(self) -> u32 {
        self.base
    }
    pub const fn race_k(self, k: usize) -> u32 {
        self.base + k as u32 * self.words()
    }
    /// Word of a param value in the params buffer.
    pub const fn param(self) -> u32 {
        self.base
    }
    pub const fn param_k(self, k: usize) -> u32 {
        self.base + k as u32 * self.words()
    }
    /// Calls `f(h, k, plane)` for every value of this field in a race with `horses` horses (`h` = None for race
    /// fields).
    pub fn for_each_plane(self, horses: usize, mut f: impl FnMut(Option<usize>, usize, u32)) {
        match self.scope {
            Scope::Param => {}
            Scope::Race => {
                for k in 0..self.len as usize {
                    f(None, k, self.race_k(k));
                }
            }
            Scope::Horse => {
                for h in 0..horses {
                    for k in 0..self.len as usize {
                        f(Some(h), k, self.horse_k(h, k));
                    }
                }
            }
        }
    }
    /// "horse 3 name[k]" / "name".
    pub fn label(self, h: Option<usize>, k: usize) -> String {
        let name = if self.len > 1 { format!("{}[{k}]", self.name) } else { self.name.to_string() };
        match h {
            Some(h) => format!("horse {h} {name}"),
            None => name,
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/layout_gen.rs"));

/// One race's state, one word per plane, with a record of which planes were written.
#[derive(Clone)]
pub struct LaneState {
    pub words: Vec<u32>,
    written: Vec<bool>,
    pub horses: usize,
}

impl LaneState {
    pub fn new(horses: usize) -> Self {
        assert!(horses <= MAX_HORSES as usize);
        LaneState { words: vec![0; STATE_PLANES as usize], written: vec![false; STATE_PLANES as usize], horses }
    }
    pub fn u32(&mut self, plane: u32, v: u32) {
        self.words[plane as usize] = v;
        self.written[plane as usize] = true;
    }
    pub fn i32(&mut self, plane: u32, v: i32) {
        self.u32(plane, v as u32);
    }
    pub fn f32(&mut self, plane: u32, v: f32) {
        self.u32(plane, v.to_bits());
    }
    pub fn bool(&mut self, plane: u32, v: bool) {
        self.u32(plane, v as u32);
    }
    pub fn f64(&mut self, plane: u32, v: f64) {
        self.u32(plane, v.to_bits() as u32);
        self.u32(plane + 1, (v.to_bits() >> 32) as u32);
    }
    /// Values of fields that should have been written and were not (a gap in `pack`).
    pub fn missing(&self) -> Vec<String> {
        let mut out = Vec::new();
        for f in ALL_FIELDS {
            f.for_each_plane(self.horses, |h, k, p| {
                if !(0..f.words()).all(|w| self.written[(p + w) as usize]) {
                    out.push(f.label(h, k));
                }
            });
        }
        out
    }
}

/// The params buffer (one word per value), with a record of which words were written.
pub struct ParamWords {
    pub words: Vec<u32>,
    written: Vec<bool>,
}

impl Default for ParamWords {
    fn default() -> Self {
        ParamWords { words: vec![0; PARAM_WORDS.max(1) as usize], written: vec![false; PARAM_WORDS.max(1) as usize] }
    }
}

impl ParamWords {
    pub fn u32(&mut self, word: u32, v: u32) {
        self.words[word as usize] = v;
        self.written[word as usize] = true;
    }
    pub fn i32(&mut self, word: u32, v: i32) {
        self.u32(word, v as u32);
    }
    pub fn f32(&mut self, word: u32, v: f32) {
        self.u32(word, v.to_bits());
    }
    /// Param values left out of `pack_params`.
    pub fn missing(&self) -> Vec<String> {
        let mut out = Vec::new();
        for f in ALL_FIELDS.iter().filter(|f| matches!(f.scope, Scope::Param)) {
            for k in 0..f.len as usize {
                if !(0..f.words()).all(|w| self.written[(f.param_k(k) + w) as usize]) {
                    out.push(f.label(None, k));
                }
            }
        }
        out
    }
}

/// Lane states interleaved for upload: word `plane * lanes + lane`.
pub fn interleave(states: &[LaneState]) -> Vec<u32> {
    let lanes = states.len();
    let mut out = vec![0u32; STATE_PLANES as usize * lanes];
    for (l, s) in states.iter().enumerate() {
        for (p, &w) in s.words.iter().enumerate() {
            out[p * lanes + l] = w;
        }
    }
    out
}

/// A value that differs between the CPU and the GPU.
#[derive(Clone, Debug)]
pub struct Mismatch {
    pub field: &'static str,
    pub label: String,
    pub cpu: String,
    pub gpu: String,
}

fn show(kind: Kind, lo: u32, hi: u32) -> String {
    match kind {
        Kind::F32 => format!("{:e} (0x{lo:08x})", f32::from_bits(lo)),
        Kind::F64 => {
            let b = lo as u64 | (hi as u64) << 32;
            format!("{:e} (0x{b:016x})", f64::from_bits(b))
        }
        Kind::I32 => format!("{}", lo as i32),
        Kind::U32 => format!("{lo}"),
        Kind::Bool => format!("{}", lo != 0),
    }
}

/// Compares the CPU state with lane `lane` of an interleaved GPU buffer (NaNs of any payload are equal).
pub fn diff(cpu: &LaneState, gpu: &[u32], lane: usize, lanes: usize, out: &mut Vec<Mismatch>) {
    let get = |p: u32| gpu[p as usize * lanes + lane];
    for f in ALL_FIELDS {
        f.for_each_plane(cpu.horses, |h, k, p| {
            let (c0, g0) = (cpu.words[p as usize], get(p));
            let (c1, g1) = if f.words() == 2 { (cpu.words[p as usize + 1], get(p + 1)) } else { (0, 0) };
            let same = match f.kind {
                Kind::F32 => c0 == g0 || (f32::from_bits(c0).is_nan() && f32::from_bits(g0).is_nan()),
                Kind::F64 => {
                    let (c, g) = (c0 as u64 | (c1 as u64) << 32, g0 as u64 | (g1 as u64) << 32);
                    c == g || (f64::from_bits(c).is_nan() && f64::from_bits(g).is_nan())
                }
                Kind::Bool => (c0 != 0) == (g0 != 0),
                _ => c0 == g0,
            };
            if !same {
                out.push(Mismatch { field: f.name, label: f.label(h, k), cpu: show(f.kind, c0, c1), gpu: show(f.kind, g0, g1) });
            }
        });
    }
}
