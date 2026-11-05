//! Times GameData load, saved-race decode and `sim::simulate` separately over a corpus (single thread).
//!   cargo run --release --bin bench -- <corpus dir> [reps] [hash out file] [data_dir]
//! Each race is simulated `reps` times (default 3) in its own arena scope; the minimum total over reps is reported. If
//! a hash file is given, one line per race with a hash of the full SimOutput (all frame/result/event bits) is written,
//! to check that a performance change leaves the output bit-identical.

use std::alloc::{GlobalAlloc, Layout};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use uma_race_sim::api::SimOutput;
use uma_race_sim::arena::{self, ArenaAllocator, ThreadArena};
use uma_race_sim::data::{DataEra, GameData, corpus};

/// Counts heap allocations (alloc + realloc calls), to spot per-frame allocations in the sim.
struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { ArenaAllocator.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { ArenaAllocator.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { ArenaAllocator.realloc(p, l, n) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn output_hash(o: &SimOutput) -> u64 {
    let mut h = DefaultHasher::new();
    for f in &o.frames {
        f.time.to_bits().hash(&mut h);
        for x in &f.horses {
            (x.distance.to_bits(), x.lane_position.to_bits(), x.speed.to_bits(), x.hp.to_bits()).hash(&mut h);
            (x.temptation_mode, x.block_front_horse_index).hash(&mut h);
        }
    }
    for r in &o.results {
        (r.finish_order, r.finish_time_raw.to_bits(), r.finish_time_scaled.to_bits(), r.finish_diff_time.to_bits()).hash(&mut h);
        (r.start_delay_time.to_bits(), r.last_spurt_start_distance.to_bits(), r.guts_order, r.wiz_order, r.running_style, r.defeat).hash(&mut h);
    }
    for e in &o.events {
        (e.time.to_bits(), e.kind, &e.params).hash(&mut h);
    }
    h.finish()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().expect("usage: bench <corpus dir> [reps] [hash out file] [data_dir]");
    let reps: usize = args.get(1).and_then(|r| r.parse().ok()).unwrap_or(3);
    let data_dir = PathBuf::from(args.get(3).map_or(uma_race_sim::data::DATA_DIR, |s| s.as_str()));
    let files = corpus::race_files(Path::new(dir));

    let t = Instant::now();
    let races: Vec<_> = files.iter().filter_map(|f| {
        let (input, _rep, date) = corpus::load_race(f).ok()?;
        Some((f, input, DataEra::for_date(&date).ok()?))
    }).collect();
    let decode = t.elapsed().as_secs_f64();

    let mut cache: HashMap<DataEra, GameData> = HashMap::new();
    let t = Instant::now();
    for (_, _, era) in &races {
        if !cache.contains_key(era) {
            cache.insert(era.clone(), GameData::load(&data_dir, era.clone()).unwrap());
        }
    }
    let load = t.elapsed().as_secs_f64();

    // Per-race minimum over the reps (robust against other processes sharing the CPU), summed.
    let mut best = vec![f64::INFINITY; races.len()];
    let mut hashes = Vec::new();
    let mut allocs = 0;
    let _arena = ThreadArena::attach();
    for _ in 0..reps {
        hashes.clear();
        for (k, (f, input, era)) in races.iter().enumerate() {
            let a = ALLOCS.load(Relaxed);
            let hash = {
                // SAFETY: The output ends with the scope, and the hash holds nothing of it.
                let _scope = unsafe { arena::open_scope() };
                let t = Instant::now();
                let out = uma_race_sim::sim::simulate(input, &cache[era]);
                best[k] = best[k].min(t.elapsed().as_secs_f64());
                allocs += ALLOCS.load(Relaxed) - a;
                output_hash(&out)
            };
            hashes.push(format!("{hash:016x} {}", f.display()));
        }
    }
    let best: f64 = best.iter().sum();
    let n = races.len() as f64;
    println!("{} races, {} eras", races.len(), cache.len());
    println!("decode (read+parse+replay, all races) {:.1} ms = {:.3} ms/race", decode * 1e3, decode * 1e3 / n);
    println!("GameData load {:.1} ms = {:.1} ms/era", load * 1e3, load * 1e3 / cache.len() as f64);
    println!("simulate (per-race min of {reps}) {:.1} ms = {:.3} ms/race", best * 1e3, best * 1e3 / n);
    println!("allocations in simulate: {:.0}/race", allocs as f64 / reps as f64 / n);
    if let Some(p) = args.get(2) {
        std::fs::write(p, hashes.join("\n") + "\n").unwrap();
    }
}
