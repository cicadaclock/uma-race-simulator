# Race Simulator Program

## Logic of the Step Cycle

### 1. General Description

The program computes a horse race as a sequence of discrete time steps. Each step advances the race clock by a fixed interval of 0.0666 second. This constant is inherited from the game and is not equal to 1/15. Processing continues until every runner has passed the goal, so a runner that has finished continues to be carried past the line until the last runner arrives.

The program is deterministic. A given input record and generator seed reproduce the same result, bit for bit. The reader is cautioned that the order of operations is part of the specification. Runners are updated one at a time in ascending gate sequence, and a runner sees the already updated state of any runner processed before it. Rearranging the sequence alters the result.

### 2. Storage Organization

Figure 1 lists the principal control blocks. Values are chiefly short floating-point (`f32`, 4 bytes, IEEE binary), with long floating-point (`f64`) used inside individual expressions where the game so computes them.

**Figure 1. Principal control blocks**

| Block | Bytes | Occurrence | Contents |
|---|---|---|---|
| `Race` | 2,304 | 1 per race | Root block: clock, generator, course state, runner table |
| `Field` | 1,024 | 1 per race | Standings, style lists, pacemaker, end-of-step caches |
| `ScanView` | 320 | Within `Field` | Start-of-step columns of distance, lane and speed |
| `RaceRandom` | 248 | 1 per race | Generator state |
| `Horse` | 1,576 | 1 per runner | Live state of the runner, including `Ai` (424) |
| `AroundTable` | 704 | 1 per runner | Gaps to every other runner, one column per value |
| `SkillWalk` | 2,400 | 1 per runner | Wake thresholds of the runner's skills |
| `SkillBase` | 288 | 1 per skill | Skill state, with two `SkillDetail` entries (128 each) |
| `SkillTrigger` | 48 | 1 per condition term | One term of a skill condition |
| `SkillParamModifier` | 24 | 1 per active effect | Kind, value and time remaining |
| `HorseFrame` | 28 | 1 per runner per frame | Recorded output values |

Working storage is at least 10 KB per runner on average, or about 120 KB for a field of twelve. Race construction requests approximately 546 KB in total, chiefly for course tables and skill data.

The items fall into four classes:

- **Continuous quantities:** distance, speed, HP, lane and timers.
- **Discrete state:** phase, position keep mode, order, counters and flags. At its top level, `Horse` declares 52 `f32`, 32 `i32` and 14 `bool` items.
- **References:** runners are referred to by gate index, not by address.
- **Variable lists:** skills, modifiers, overtake targets and events.

**Figure 2. Observed values, 963 races (15.4 million runner steps)**

| Item | Lowest | Mean | Highest |
|---|---|---|---|
| Course distance per race (m) | 1,000 | 1,888 | 3,600 |
| Runners per race | 9 | 11.6 | 12 |
| Steps per race | 712 | 1,399 | 2,948 |
| Finish time (s) | 46.2 | 92.2 | 196.3 |
| Skills per runner | 8 | 14.5 | 33 |
| Condition terms per runner | 13 | 32.3 | 87 |
| Speed (m/s) | 3.00 | 20.57 | 27.15 |
| Target speed (m/s) | 0.00 | 21.03 | 30.00 |
| Acceleration (m/s²) | 0.02 | 0.60 | 25.22 |
| Distance moved per step (m) | 0.00 | 1.37 | 1.81 |
| HP | 0 | 1,720 | 4,543 |
| Lane position | 0.000 | 0.131 | 0.698 |
| Active modifiers per runner | 0 | 2.1 | 14 |
| Near horses per runner | 0 | 1.6 | 9 |
| Generator draws per step | 0 | 0.22 | 19 |

Lane position is dimensionless, with adjacent gates 1/18 of a unit apart. Of all runner steps, 28.8% were in last spurt, 17.6% in a position keep mode and 2.8% blocked by a runner ahead.

### 3. Processing of One Step

The step routine adds the interval to the race clock and invokes the frame routine ([tick.rs:77](src/sim/tick.rs#L77)), which performs the following stages in the order given.

1. **Snapshot.** Distance, lane and speed of all runners are copied into `ScanView`.
2. **Pre-update, each runner.** The runner's `AroundTable` is filled with distance, lane and speed gaps, which determine vision, near horses and blocking. Modifier timers are decremented and expired modifiers removed.
3. **Skill processing, each unfinished runner.** Skills of priority 1 are processed in one pass and skills of priority 0 in a second. See Section 5.
4. **Runner update, each runner.** Decisions are made, then the runner is moved. See Section 4.
5. **Field resolution.**
    - Runners that overlap by less than 0.4 m and 0.0222 lane units are pushed apart.
    - The spot struggle and duel rules are then evaluated.
    - The pacemaker is then selected.
6. **Post-update, each runner.** Near-horse timers are advanced.
7. **Order.** The standings are sorted by distance using .NET sort semantics. Upon the final step, finish order is computed instead.
8. **End-of-step caches.** The caches of rushed horses, activated skills and lane rates are refreshed.

Upon return, the step routine dispatches course events (corners, straights, slopes, lane limits), records state spans and writes a frame if one is due.

### 4. Runner Update

The decision routines are performed in this sequence: position keep, overtake, rush, target lane, last spurt, downhill mode, conserve power and target speed.

Movement is then computed as follows:

```text
target  = section base speed or last spurt speed
          x position keep multiplier + skill, slope, struggle and duel terms
          limited to the range [minimum, 30]
accel   = sqrt(power x coefficient) x slope, ground, distance and phase factors
          + skill, duel and conserve power terms + start dash term
speed   = moves toward target by accel x dt, or falls at the phase deceleration rate
v       = speed + current speed modifiers, limited to the minimum and maximum speeds
moved   = v x dt, limited by displacement along the course path
HP      = HP - rate(v, state) x dt, with a guts factor late in the race
```

Each of the 24 course sections has a base target speed, with a random component fixed for each runner before the first step. Last spurt is computed upon entering the End phase, and again upon a late heal. The routine searches in 30 m increments for the longest spurt that remaining HP can sustain, then reduces speed if none fits. A runner's finish time is interpolated back to the moment it crossed the line.

World position is obtained by interpolating a course path table of 1,001 keys, each holding a position and a rotation quaternion. Its sole purpose is the displacement limit above.

### 5. Skill Processing

A skill condition is parsed into OR groups of AND terms, stored contiguously per runner. There are approximately 126 forms of term, such as distance rate, order or corner. Each term compares a value to a constant.

So that dormant skills consume no time, `SkillWalk` holds the phase and distance at which each skill could next become relevant, and is tested eight entries at a time with AVX2. Only awake skills are examined.

When a skill activates, its abilities select target runners and place modifiers upon them. All later computation reads skill effects through 26 modifier parameters, applied in insertion order. For example, the winner in Figure 4 holds `Speed: Add +40` with no expiry.

### 6. Random Number Generator

The generator is the .NET Framework `System.Random`, a Knuth subtractive generator of 56 fullwords. One generator serves the entire race, and its sequence defines the result. In the sample race, construction used 162 draws, the start routine 368 and all 1,861 steps only 308. Draws are therefore infrequent during the race, but each is significant.

### 7. Output

Four records are produced:

- **Frames:** every step during the first second and within 25 m of the goal for a player's runner, otherwise once per second. A race averages 140 frames.
- **Events:** time, kind and a parameter list, averaging 110 per race. Example: time 0.0, kind 3 (skill), parameters `[0, 200012, -1, 0, 0, 0]`.
- **Spans:** state spans and skill spans, averaging 142 and 104 per race.
- **Results:** finish order, finish times, start delay and defeat reason for each runner.

### 8. Sample Execution

The sample is course set 10506, 2,500 m, nine runners, base speed 19.5 m/s. The winner is gate 5, an Oikomi runner with raw stats of 1439 speed, 903 stamina, 977 power, 719 guts and 778 wit, maximum HP 3,204.4, 20 skills and 67 condition terms. Each line shows state upon entry to the numbered step.

**Figure 3. Winner, step by step**

```text
 step    time distance      phase   speed  target  accel      HP   lane ord      keep mods draws
    0   0.000     0.00      Start   3.000   0.000 24.382  3204.4  0.222   5      Null    1   530
    2   0.133     0.28      Start   4.624  16.726 24.382  3203.6  0.222   8  PaceDown    1   537
  200  13.320   227.40      Start  18.298  18.926  0.382  3098.4  0.090   9    PaceUp    1   557
  400  26.640   475.12  MiddleRun  19.680  19.696  0.404  2866.8  0.161   9      Null    2   588
  800  53.280   998.42  MiddleRun  19.559  19.906  0.269  2366.0  0.142   8    PaceUp    1   657
 1200  79.920  1525.55  MiddleRun  20.039  20.432  0.404  1945.7  0.000   5      Null    3   826
 1400  93.239  1806.10        End  23.574  24.634  0.403  1544.8  0.090   5      Null    1   827
 1600 106.559  2132.34  LastSpurt  24.597  25.974  0.603   834.4  0.291   4      Null    8   837
 1819 121.143  2498.89  LastSpurt  24.784  24.784  0.403     6.2  0.490   1      Null    2   837
 1861 123.940  2567.86   Finished  24.784  24.784  1.000     2.5  0.490   1      Null    2   838
```

*Note:* The acceleration of 24.382 includes the start dash term. The winner crossed the line at 121.19 s with 6 HP remaining, and processing continued until the last runner finished at 123.89 s.

**Figure 4. Entire field at step 909**

```text
gate style   distance  speed  target    lane      HP order blocker near mods
   1 Nige     1177.33 19.681  19.681  0.0143  2015.1     1       -    1    4
   2 Sashi    1144.18 19.494  19.494  0.0000  2149.6     5       -    2    0
   3 Nige     1175.19 19.637  19.637  0.0000  1974.5     2       -    1    3
   4 Sashi    1142.47 19.460  19.977  0.0085  2033.3     6       2    3    0
   5 Oikomi   1141.62 19.519  19.519  0.1345  2212.4     7       -    3    1
   6 Nige     1167.69 19.526  19.549  0.0000  2141.7     3       -    0    0
   7 Sashi    1132.81 19.563  19.882  0.0000  2205.4     9       -    0    1
   8 Sashi    1140.94 19.405  20.915  0.0234  2313.4     8       4    2    2
   9 Oikomi   1150.23 19.878  19.554  0.0000  2120.7     4       -    0    2
```

Gate 8 illustrates blocking: it is held to 19.405 m/s behind gate 4 despite a target of 20.915 m/s. At this instant gate 5 records gate 4 as 0.84 m ahead and 0.1272 lane units inward, with a speed gap of +0.059 m/s.

## GPU Compute Shader Implementation

Currently the program is implemented on the CPU. The general performance is about 10ms per race simulated. This inherently limits the amount of races we can simulate to ~800 races/s on 12 threads, due to CPU/cache contention. Your task is to reason about a hypothetical GPU implementation that can simulate races across the more parallel GPU whilst retaining accuracy. A naive GPU implementation got relatively close, however the accuracy of floating point calculations changed a lot of the in-step results of the race, making the result useless.

### Naive Implementation

**Seed by seed, it no longer matches the game (963 races):**

| Field update | Same finish order | Same winner | Average finish time change |
|---|---|---|---|
| Separate generators only | 0.1% | 45.6% | 0.276 s |
| Double buffer, shared generator | 1.0% | 48.8% | 0.257 s |
| Double buffer, separate generators (your design) | 0.0% | 42.9% | 0.280 s |
| Exact sim, different seed | 0.0% | 33.6% | 0.346 s |

Separate generators alone change 99.9% of finish orders. The game draws from one generator in gate order, and how many numbers a horse draws depends on earlier horses in that step, so this can't run in parallel. Replays and anything else that must match the game stay on the CPU.

**Across many seeds, it stays close (161 races, 200 seeds each):**

| Compared with exact | Win rate change | Finish time bias | Same favorite |
|---|---|---|---|
| Exact sim, new seeds (normal variation) | 1.20 points | +0.0005 s | 85.7% |
| Separate generators only | 1.24 points | +0.0003 s | 87.6% |
| Double buffer, shared generator | 1.47 points | +0.022 s | 85.7% |
| Your design | 1.53 points | +0.021 s | 85.1% |

- **Generators:** separate generators alone stay within normal variation.
- **Double buffer:** it adds a small real bias. Runners finish 0.02 s slower, and Senko's win share drops from 44.2% to 43.4%.

**Skill rankings mostly survive (54 skills, 20 races, 40 seeds, by finish time):**
- **Rank correlation with exact:** 0.928, against 0.963 for the exact sim with new seeds.
- **Top 10:** identical in every comparison.

#### Speed

I timed toy kernels on the RX 7900 XT against all 12 CPU threads, with 18,620 races of 15 runners over 1,398 steps:

| Step type | CPU | GPU | GPU vs CPU |
|---|---|---|---|
| Physics: field scan, speed, move, HP | 10,466 races/s | 467,212 races/s | 44.6× |
| Skill conditions with a `switch` | 8,186 races/s | 17,189 races/s | 2.1× |
| Skill conditions from a table of facts, no branching | 7,682 races/s | 50,036 races/s | 6.5× (6.1× the CPU's faster `switch` version) |

A physics engine comparison holds. The catch is that skill and AI logic take about 70% of a real step, so that row decides the total. As a rough estimate:

| If real logic behaves like | Whole-sim speedup |
|---|---|
| The table kernel | About 8× |
| The `switch` kernel | About 3× |
| Only as fast as the CPU | About 1.4× |

These toy kernels do about a ninth of the real work per race, and real lane and overtake logic is much messier, so I wouldn't count on the high end.
