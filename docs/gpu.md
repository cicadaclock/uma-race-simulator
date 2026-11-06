# GPU backend (Vulkan GLSL)

Status: the device layer, the bit-exact arithmetic library and `System.Random` pass the self-test on an RX 7900 XT.
Every stage of the race step is ported, each checked against the CPU on real race states, and whole races run on the
GPU (`gpu::run`): 100 synthetic races match the CPU at every one of their steps, race state and skill state, with
`gpu_racecheck` (see below), on lavapipe and on the RX 7900 XT, and 16384 seeds of a lobby give the CPU's results on
the 7900 XT with `gpu_bench`. Not yet ported: the pre-July-2026 pacemaker, and observed / muted skills.

| Stage (`tick::Stage`) | Ported | Checked on (lavapipe) |
|---|---|---|
| Start of the step (`tick::begin_step`) | the race clock (the recorder's clock is output only) | 16 k samples, 10 synthetic races |
| PreUpdate | all: scan view, around table, near / visible / duel-candidate lists, blocking, lane space, modifier timers | 146 k samples, 100 synthetic races |
| SkillsFirst, SkillsSecond | all of the per-frame update: condition terms, preconditions, cooldowns, activation, the 23 target types, ability values and durations, additional activations, time extensions, the random reservation (ability types 36 / 37). Not observed or muted skills (see below) | 143 k samples each, 100 synthetic races, and 31 k each on 20 more. Rare paths: see `--coverage` below |
| UpdateHorses | all: min / max / target speed, acceleration, HP, position keep, rushing, overtaking, lane planning and moves, last spurt, downhill mode, conserve power, the course path, block and order timers, the finish time. `skill::clear_skills` at the finish | 161 k samples with the skill state, 100 synthetic races, perturbed 1 in 8. Before the skills: 165 k samples, 100 synthetic races, 18 k of them perturbed (see below). Synthetic races use the current data, so the pre-July-2026 branches run only with `--corpus` |
| Field | all, for the July 2026 courses on (before that, the pacemaker update is in EndOfFrame) | 165 k samples, 100 synthetic races, 18 k of them with a horse moved onto another (`--perturb`) |
| PostUpdate | all | 146 k samples, 100 synthetic races |
| Order | all, including the last frame's finish order and defeat reasons | 146 k samples, 100 synthetic races |
| EndOfFrame | all but the pre-July-2026 pacemaker, including the skill caches (`prev_activated_skills`, the heal history) | 143 k samples with the skill state, 100 synthetic races |
| End of the step (`tick::end_step`) | the distance spread and the course events (ground, straights, corners, lane width, slopes). Not the recording | 143 k samples, 100 synthetic races |

Shared pieces:

- **Skills** (`gpu::skills`, `shaders/lib/skill_state.glsl`, `shaders/lib/skills.glsl`): two more buffers, see below.
- **Skill modifiers** (`shaders/lib/modifiers.glsl`). The Rust keeps one list per parameter. The GPU keeps one pool per horse ordered by parameter, with each parameter's entries in list order. That is the only order any operation depends on, and `gpu::pack` writes the same order. The pool holds 24 entries (14 is the most in 963 replays); overflow sets `gpu_fault`.
- **The race RNG** is part of the race state (the `rng` field: seed array, `inext`, `inextp`, draw count), so a stage that draws a different number of times, or in a different order, shows up as a mismatch.
- **The course path** (`shaders/lib/course_path.glsl`) is a third buffer of per-course blocks (`gpu::pack::course_words`: the lane animation, the overrun animation and the goal frame); each race's `course_base` points at its block. The GPU reproduces `lanes::calc_world_transform` bit for bit, including the f64 `%` of the overrun loop.
- **`Mathf.Pow` / `Mathf.Log10`** (`shaders/lib/transcend.glsl`). Computed in double-double (about 2^-96), so the f64 value before the final rounding is the correctly rounded one. Every input comes from a stat clamped to [1, 2000], so `gpu_mathcheck` tries all 92 M f32 stats at each of the 8 call sites against the CPU's libm. On lavapipe against glibc, all 735 M match, and 40 k random f64 results are correctly rounded against exact (Python `decimal`) values. A CPU libm that misrounds near an f32 boundary shows up as a mismatch; those inputs would go in an exception table. The HP formula's `Pow(x, 2)` takes a speed difference, not a stat; the GPU uses `x * x`, and `gpu_mathcheck` checks the CPU's `Pow(x, 2)` against that for every f32 in [-256, 256].

## Design

**Parallel across races, never within one.** Each lane of a wave runs one whole race in the CPU's order: entry
order, one `System.Random` per race, the .NET introsort. Nothing is double-buffered and no generator is split, so a
GPU race can match `sim::` seed for seed. All lanes of a wave run the same lobby (course, runners, skills) with
different seeds, so the loops over runners, skills and condition terms branch the same way in every lane.

**Float semantics** are what remains, and they are handled in four places:

| Where | What |
|---|---|
| `src/gpu/spirv.rs` | At load time, every float op in the SPIR-V gets `NoContraction` (no fused multiply-add), and the entry point gets the float-controls modes the device supports: denormals preserved, signed zero / Inf / NaN preserved, round to nearest even. This does not depend on writing `precise` in the GLSL. |
| `shaders/lib/exact.glsl` | The `sim::util` helpers with the game's NaN and signed-zero behaviour, and `/` and `sqrt` in f32 and f64 (`div_rn32`, `div_rn64`, `sqrt_rn32`, `sqrt_rn64`), correctly rounded for every input, subnormals and specials included. Vulkan only promises 2.5 ulp for `/`. On the RX 7900 XT (driver 26.8.1) the hardware misrounds 30% of f32 quotients, 26% of f64 quotients and 15% of f32 roots, and is far off for subnormals. So the hardware result is only a starting guess, and a guess that doesn't settle is replaced by one from the exact f64 routines. |
| `Gpu::new` | Probes whether `fma()` rounds once. Vulkan allows two roundings, and lavapipe does round twice. The answer goes to every kernel as specialization constants: with a fused fma the refinements use fma residuals; without, exact f64 residuals (f32) and Dekker products (f64). |
| `gpu_selftest` | Runs all of the above on the device and compares bit for bit with the Rust functions the CPU sim uses. |

**Skills.** The skill program (binding 3) is built on the CPU once per race and never changes: the parsed
condition terms, the abilities with their value and duration tables, the runners' fixed data, and where each piece
of state lives. The skill state (binding 4) is what changes: per detail (`is_activated`, time left, cooldown, ...),
per ability (the value on activation, additional activations, the modifier ids it pushed), per stateful term, and
the lists (used ids, active details, last frame's details, reserved ids, the heal history). The record formats are in
`src/gpu/skill_spec.rs`, and build.rs generates the GLSL and Rust constants from it. Whatever the CPU decides while
building a skill (parsing, the windows and lots drawn at construction) stays on the CPU and is packed.

The CPU skips skills and conditions that cannot fire yet (the skill walk, `is_dormant`, the wake thresholds). The GPU
updates everything every frame. That changes only the `is_activated` of terms without state, which nothing reads, so
the GPU keeps those only where they are read (`change_order`, and collections with `random_lot_other_activate`).
An ability's modifiers are kept as runs of consecutive modifier ids (6 runs at most; more sets `gpu_fault`). Not
supported: observed skills (a skill with its own generator, for studying one skill's effect) and muted ones. The
program flags them and every kernel that reads skills faults.

## Running

```sh
cargo run --release --features gpu --bin gpu_mathcheck              # Pow / Log10 at every call site, every stat
cargo run --release --features gpu --bin gpu_stagecheck             # ported stages against the CPU
cargo run --release --features gpu --bin gpu_stagecheck -- --perturb  # also paths real races rarely reach
cargo run --release --features gpu --bin gpu_stagecheck -- --stage order --races 500 --seed 7
cargo run --release --features gpu --bin gpu_stagecheck -- --corpus "path/to/Saved races"
cargo run --release --features gpu --bin gpu_stagecheck -- --stage skills_second --explain  # every value of the first bad sample
cargo run --release --features gpu --bin gpu_stagecheck -- --coverage  # which skill paths the races reach (CPU only)
cargo run --release --features gpu --bin gpu_racecheck              # whole races, compared with the CPU every step
cargo run --release --features gpu --bin gpu_racecheck -- --races 20 --seeds 64 --chunk 100  # 64 seeds per lobby
cargo run --release --features gpu --bin gpu_selftest              # first discrete GPU
cargo run --release --features gpu --bin gpu_selftest -- --no-fma   # also test the non-fma paths
cargo run --release --features gpu --bin gpu_selftest -- --list     # devices; choose with UMA_GPU=<index|name>
cargo run --release --features gpu --bin gpu_selftest -- --scale 8 --seed 7
cargo test --release --features gpu --lib gpu::                     # SPIR-V patch + CPU copies of the fma paths
```

`ash` loads `vulkan-1.dll` at run time, so building needs no Vulkan SDK. `build.rs` compiles `shaders/*.comp` with
`glslangValidator` when it finds one (`GLSLANG_VALIDATOR`, `%VULKAN_SDK%\Bin`, or `PATH`). Otherwise it uses
`shaders/prebuilt/*.spv` and prints a warning. Regenerate those after editing a shader or the state layout:

```sh
UMA_UPDATE_PREBUILT=1 cargo build --release --features gpu
```

`gpu_selftest`: exit status 0 means every required row passed. Rows with subnormal inputs or results are counted separately: a
device that flushes denormals fails only those, which matters only if the sim produces subnormals.

`gpu::exact_ref` runs a CPU copy of the division and square-root code with and without a fused fma. It uses a
simulated device whose `/` and `sqrt` are only as accurate as Vulkan requires, and garbage for subnormals, and
checks 3.6 M full-range cases against Rust. Lavapipe (Mesa's CPU Vulkan driver) passes the self-test in both modes.
Its `fma()` is not fused and its `/` and `sqrt` are exact, so a real GPU is needed to cover the fused paths and the
fallbacks from bad hardware guesses.

## Stage-by-stage checking

`tick::run_stage` runs one stage of a frame on the CPU (`frame` is now the eight stages in a row, with
output identical to before); `begin_step` and `end_step` are the two other checked points. `gpu_stagecheck` plays each race on the CPU. At
the stage under test it packs the state just before the stage into one GPU lane, runs the stage there, and compares
every value of the lane with the CPU state just after the stage. A mismatch names the race, the step, the field and
both values. Kernels that touch skills also get the skill program and state, and every skill state word is compared
too.

Races come from `synth::lobby` (random stats, styles, aptitudes and 6-30 skills on any race instance whose course
files are present) and, with `--corpus`, from saved races.

**The state layout** lives in `src/gpu/layout_spec.rs`, one line per field. `build.rs` generates the GLSL
accessors (`race_state.glsl`: `h_distance(h)`, `set_h_distance(h, v)`, `r_order_horse(k)`, ...) and the Rust plane
constants (`gpu::layout::H_DISTANCE`, ...) from it. Each value is a plane of `lanes` words with the lane fastest, so
a wave reads consecutive words. `gpu::pack` fills a lane from a `Race`, and `gpu_stagecheck` stops if any field of
the list was left out. To port a stage:

1. Add the fields it reads or writes to `layout_spec.rs` and to `gpu::pack`.
2. Port the Rust to `shaders/lib/<stage>.glsl` (rules below) and add a `shaders/stage_<name>.comp` kernel.
3. Add it to `PORTED` in `src/bin/gpu_stagecheck.rs` and to `gpu::shaders`.

A kernel that reaches a path it doesn't implement sets `r_gpu_fault`, so the sample shows up as a mismatch.

Some paths almost never occur in real races; the overlap push-apart never fired in 582 k synthetic frames. A
`perturb` function on a `PORTED` entry changes a copy of the state to reach such a path. With `--perturb`, one
sampled frame in eight is also checked on a perturbed copy.

- Field: one horse put on top of another.
- UpdateHorses: on a random running horse, one of a rush starting now, a position-keep check now (sometimes with no
  pacemaker yet), a position-keep mode in progress (often on the leader or the pacemaker), another pacemaker, a skill
  modifier few skills give (target lane, forced overtaking side, HP rate, target speed while changing lanes, rush
  end), a last-spurt recompute at a random HP and Wit (and a front blocker low on HP), or every Nige but one turned
  Oonige with one of them in SpeedUp.

`--coverage` runs the races on the CPU only and counts, for each feature of a skill detail (trigger kind, ability
type, target type, value and time usage, preconditions, OR-groups, ...), the details that have it, how many are
active after `tick::start`, and how many activations they had in later steps. Over 200 synthetic races, every target
type, value usage and time usage activates in a step at least 10 times, and so do most trigger kinds (`grade`,
`straight_front_type` and `lastspurt` 4 to 9 times). Twelve kinds never activate in a step: passive conditions
(weather, season, ground condition, post number, ...) that are settled from the start, so their skills are active
from the zero-DT first frame or never. A bug in the `lastspurt` mapping passed 20 races and showed up in 100.

When a stage passes on the first try, plant a bug in each new path and check that `gpu_stagecheck` reports it.
Every path so far has been tested this way (130 planted bugs in UpdateHorses). These UpdateHorses lines were never
reached, so a bug in them would go unseen:

- `last_spurt.glsl`: the fallback to the last candidate when every candidate fails its Wit check.
- `course_path.glsl`: the extrapolation past the goal on a course without an overrun path (every course in `data/`
  has one), and a key time of exactly 1000.
- `temptation.glsl`: the Null rush mode of Senko and Oikomi (their percentages add up to 100).

The skills got 37 planted bugs (`skills_second` on 8 races, the skill state on 20, then the misses again on 300 whole
races with `gpu_racecheck`), and 29 were caught. The other 8 are in code the current data never reaches, or they
change nothing:

- No skill uses `random_lot_other_activate` or `change_order` (two bugs in the first, one in the second), and none
  has rarity 6 (the random reservation's rarity filter).
- `same_skill_horse_count` counts in the zero-DT first frame, before any checked step.
- `ActivateAnySkill`'s "own detail" case: a skill's own entries leave the previous-frame list before its update.
- `all_corner_random`: the windows are in ascending order, so stopping at the first window ahead equals skipping on.
- The debuff filter for teammates: synthetic lobbies have no teams (corpus team races do).

The end of the step got 14, and 9 were caught. A missing ground change took 300 races to show. The other 5
change nothing on the current courses: the spread on the last step (no horse is running), the straight's start
test and the `was_straight` test (straights don't overlap), and the fallbacks for an event with no matching corner
or slope.

## Whole races

`gpu::run` runs races from start to finish. The CPU builds each race (`race::init`, `tick::start`: the skill windows
and lots drawn at construction, the zero-DT first frame) and packs it into a lane (`LaneInput`). A step is ten
kernels in a row (`STEP_KERNELS`: begin_step, the eight frame stages, end_step), and `Runner::steps` records any
number of steps as one submission, with a memory barrier after each dispatch (`steps_paced` splits them so a
submission takes about 250 ms, well under the 2 s after which Windows resets the GPU). On a race's last step the end_step
kernel sets `race_done`, and every kernel then returns at once for that lane, so a batch runs until its longest race
ends and each lane stops in the CPU's final state.

`gpu_racecheck` steps the GPU lanes and CPU copies of the same races in lockstep and compares every value of every
lane (race state and skill state) every `--chunk` steps (default 1). `--seeds N` runs each lobby with N race seeds
next to each other, the intended use: lanes of one wave share the lobby, so they branch alike.

`gpu_bench` measures throughput. It runs one lobby with many race seeds, on every CPU thread (the CPU sim, with its
arena allocator) and on the GPU: the CPU builds and packs each race on every thread, then the GPU steps them all,
and the next batch is built while the GPU runs one. It prints each part's rate and checks every race's results (each
runner's finish order and finish time) against the CPU. `--profile` times each kernel (timestamps over the first 200
steps).

```sh
cargo run --release --features gpu --bin gpu_bench -- --seeds 65536 --batch 16384 --profile
cargo run --release --features gpu --bin gpu_bench -- --race "path/to/a race.json" --seeds 65536 --batch 16384
```

The CPU's share of a GPU race is building it: `race::init` and `tick::start` (about 300 µs for 10 runners on one
thread) and packing (about 170 µs), against 10 ms for the whole race on the CPU. That caps the speedup at about 20
times all the CPU threads, however fast the GPU. A lane holds 53 KB of race state and the skill state (tens of KB,
by lobby); `gpu_bench` prints both. The course block and the skill program (60 KB and 90 KB) are kept once per
batch (`run::Shared`): all seeds of a lobby share them.

First measurement on the RX 7900 XT (10 runners, 16384 seeds in one batch, 12 CPU threads): the CPU ran 676
races/s, and the GPU's steps ran 1533 races/s (2.3 times), with all results identical. End to end it was 465
races/s, mostly because the upload took 20 s. Most of that was CPU work: every lane carried its own copy of the
program and the course (2.4 GB for the batch), the programs were hashed to find duplicates, and the transpose wrote
one word at a time. The kernels also took 261 s to build. Now the lanes share courses and programs, the transpose
works in blocks on every thread, and the kernels compile on one thread each, into a pipeline cache kept between runs
(`UMA_GPU_PIPELINE_CACHE`: a path, or `off`; by default in the temp directory).

## Porting rules for the race kernels

- One expression in Rust becomes one expression in GLSL, with the same f32 / f64 casts.
- Every `/` becomes `div_rn32` / `div_rn64`, and every `sqrt` becomes `sqrt_rn32` / `sqrt_rn64`. Never use bare `/`, `sqrt`, or `fma()` in sim code.
- Use only the `mathf_*`, `f32_to_i32`, `f64_to_i32` and `floor_to_int` helpers, never GLSL `min`, `max`, `clamp`, `round` or `int()` on values that may be out of range.
- Write constants that aren't exactly representable as bits (`uintBitsToFloat(0x3F19999Au)` for `0.6f32`). GLSL literal parsing may round differently.
- `Mathf.Pow(x, 2)` becomes `mathf_pow2` (exact). The other `Mathf.Pow` / `Log10` calls take stat values. Precompute them on the CPU per runner where the stats are fixed; otherwise port an f64 routine and test it exhaustively over the stat range.
- Don't reorder anything: no parallel reductions, the same draw order, the same loop order.
- Keep each dispatch well under 2 s. Windows resets a GPU that stays busy longer (TDR).
- Call a large function from one place. The driver inlines every call, so each call site is a full copy. The skill
  update called `value_lh` from 8 places inside 4 copies of the condition update, plus 25 `push_modifier` calls.
  Giving each one call site (a `gate` before one `value_lh` call, one loop over precondition and condition, one loop
  over (pass, detail), one `push_modifier` call) cut lavapipe's compile from 100 s to 7 s per skill kernel, and
  its memory from 3 GB to 0.6 GB. The SPIR-V barely changed, because the growth only shows after inlining: to
  estimate it, multiply each function's size in `spirv-dis` output by its copies (the product of the
  `OpFunctionCall` counts along every path from `main`).
- GLSL has no recursion: write recursive Rust (like the introsort in `order.glsl`) with an explicit stack, in the
  same order.
- Give a long dispatch one exit: a result variable, `break`, and one `return` at the end. Lavapipe (Mesa 25.2.8)
  returned the wrong case from `value_lh`, a 90-way dispatch on the term kind with a `return` per case, whether it
  was written as an `if` chain or as a `switch`. The SPIR-V was correct. The stage check is what finds such bugs, so
  run it on every driver you use.

## Next

1. Speed on the RX 7900 XT: the per-kernel profile (`gpu_bench --profile`), then the slowest kernels. The GPU
   updates every skill every frame, where the CPU skips skills that cannot fire yet (on lavapipe SkillsSecond is
   more than half the step).
2. UpdateHorses is now the largest kernel (lavapipe compiles it in 17 s): the bit-exact f64 division is inlined at
   46 call sites.
3. The pre-July-2026 pacemaker (EndOfFrame instead of Field), for the older corpus races.
