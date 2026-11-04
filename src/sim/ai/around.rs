//! The around-horse scan (vision, near, blocking, surround, compete-fight candidates), the block and near-horse
//! timers, and the queries on what the scan found.

use crate::data::ParamDefine;
use crate::sim::defs::SkillModifierParam;
use crate::sim::field::horse_by_order;
use crate::sim::horse::Horse;
use crate::sim::race::Race;
use crate::sim::util::max;

/// Most horses in a race that the scan tables hold.
pub const MAX_HORSES: usize = 24;

/// Lanes of one vector.
const LANES: usize = 8;

/// One value per horse, by horse index, aligned for vector access.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default)]
pub struct HorseColumn(pub [f32; MAX_HORSES]);

/// What one horse sees of another, from the start-of-frame scan: one entry of an [`AroundTable`].
#[derive(Clone, Copy, Debug, Default)]
pub struct AroundHorse {
    /// Horse index.
    pub info: usize,
    pub distance_gap: f32,
    pub distance_gap_abs: f32,
    pub lane_gap: f32,
    pub lane_gap_abs: f32,
    pub is_back: bool,
    /// Self - other (sign reversed vs the other gaps).
    pub speed_gap: f32,
    pub speed_gap_abs: f32,
    pub compete_fight_candidate_continue_time: f32,
}

/// What one horse sees of every other horse, from the start-of-frame scan, one column per value (perf: the scan fills
/// every column for every horse, every frame). The entry of the horse itself holds nothing of use.
#[derive(Clone, Debug, Default)]
pub struct AroundTable {
    pub distance_gap: HorseColumn,
    pub distance_gap_abs: HorseColumn,
    pub lane_gap: HorseColumn,
    pub lane_gap_abs: HorseColumn,
    /// Self - other (sign reversed vs the other gaps).
    pub speed_gap: HorseColumn,
    pub speed_gap_abs: HorseColumn,
    pub compete_fight_candidate_continue_time: HorseColumn,
    /// A bit per horse: more than the side block distance behind.
    pub back: u32,
    /// A bit per horse: every horse but the one that this table belongs to.
    pub others: u32,
}

impl AroundTable {
    /// The table of horse `i` in a race of `n` horses.
    pub fn new(n: usize, i: usize) -> Box<AroundTable> {
        assert!(n <= MAX_HORSES, "the scan tables hold {MAX_HORSES} horses, not {n}");
        Box::new(AroundTable { others: ((1u32 << n) - 1) & !(1u32 << i), ..Default::default() })
    }

    /// The entry of horse `j`.
    #[inline]
    pub fn horse(&self, j: usize) -> AroundHorse {
        AroundHorse {
            info: j,
            distance_gap: self.distance_gap.0[j],
            distance_gap_abs: self.distance_gap_abs.0[j],
            lane_gap: self.lane_gap.0[j],
            lane_gap_abs: self.lane_gap_abs.0[j],
            is_back: self.back >> j & 1 != 0,
            speed_gap: self.speed_gap.0[j],
            speed_gap_abs: self.speed_gap_abs.0[j],
            compete_fight_candidate_continue_time: self.compete_fight_candidate_continue_time.0[j],
        }
    }

    /// The index of every other horse, in horse order.
    pub fn other_horses(&self) -> impl Iterator<Item = usize> {
        let mut left = self.others;
        std::iter::from_fn(move || {
            (left != 0).then(|| {
                let j = left.trailing_zeros() as usize;
                left &= left - 1;
                j
            })
        })
    }
}

/// The horses inside the vision cone.
#[derive(Clone, Debug, Default)]
pub struct Visibility {
    /// Horse indices; the first `visible_horse_count` are valid.
    pub visible_horses: Vec<usize>,
    pub visible_horse_count: i32,
}

/// What the around-horse scan reads of each horse, one column per value.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScanView {
    pub distance: HorseColumn,
    pub lane_distance: HorseColumn,
    pub last_speed: HorseColumn,
    /// A bit per horse: on the final straight.
    pub is_straight_last: u32,
}

/// Fills `field.scan_view` from every horse. The fields are only written while horses move or by course events, so
/// one copy taken before the pre-update loop is exact for every scan in it.
pub fn snapshot_scan_view(race: &mut Race) {
    let view = &mut race.field.scan_view;
    view.is_straight_last = 0;
    for (j, h) in race.horses.iter().enumerate() {
        view.distance.0[j] = h.distance;
        view.lane_distance.0[j] = h.lane_distance;
        view.last_speed.0[j] = h.last_speed;
        view.is_straight_last |= (is_straight_last(h) as u32) << j;
    }
}

/// The limits of the scan's comparisons, from the race parameters and the horse's vision.
struct ScanLimits {
    surround_side_distance: f32,
    surround_side_lane: f32,
    surround_distance: f32,
    surround_lane: f32,
    near_distance: f32,
    near_lane: f32,
    compete_fight_lane: f32,
    compete_fight_distance: f32,
    side_block_distance: f32,
    side_block_lane: f32,
    front_block_distance: f32,
    vision: f32,
}

/// A bit per horse for each comparison of one horse against every horse.
#[derive(Clone, Copy, Default)]
struct ScanMasks {
    surround_out: u32,
    surround_ahead: u32,
    surround_behind: u32,
    near: u32,
    compete_fight_lane: u32,
    compete_fight_far: u32,
    back: u32,
    front: u32,
    side: u32,
    lane_outside: u32,
    lane_inside: u32,
    side_near: u32,
    seen: u32,
}

/// Fills the gap columns and returns the comparisons, one horse at a time.
fn scan_lanes(view: &ScanView, table: &mut AroundTable, i: usize, n: usize, limits: &ScanLimits) -> ScanMasks {
    let (my_distance, my_lane, my_speed) = (view.distance.0[i], view.lane_distance.0[i], view.last_speed.0[i]);
    let mut masks = ScanMasks::default();
    let l = limits;
    for j in 0..n {
        let distance = view.distance.0[j] - my_distance;
        let distance_abs = abs_keep_neg_zero(distance);
        let lane = view.lane_distance.0[j] - my_lane;
        let lane_abs = abs_keep_neg_zero(lane);
        let speed = my_speed - view.last_speed.0[j];
        table.distance_gap.0[j] = distance;
        table.distance_gap_abs.0[j] = distance_abs;
        table.lane_gap.0[j] = lane;
        table.lane_gap_abs.0[j] = lane_abs;
        table.speed_gap.0[j] = speed;
        table.speed_gap_abs.0[j] = abs_keep_neg_zero(speed);
        let bit = |condition: bool| (condition as u32) << j;
        masks.surround_out |= bit((l.surround_side_distance >= distance_abs) & (lane >= 0.0) & (l.surround_side_lane >= lane));
        masks.surround_ahead |= bit((distance > 0.0) & (l.surround_distance >= distance) & (l.surround_lane >= lane_abs));
        masks.surround_behind |= bit((0.0 > distance) & (distance >= -l.surround_distance) & (l.surround_lane >= lane_abs));
        masks.near |= bit(!(distance_abs > l.near_distance) & !(lane_abs > l.near_lane));
        masks.compete_fight_lane |= bit(l.compete_fight_lane >= lane_abs);
        masks.compete_fight_far |= bit(l.compete_fight_distance < distance_abs);
        masks.back |= bit(-l.side_block_distance > distance);
        masks.front |= bit((distance > 0.0) & (distance < l.front_block_distance));
        masks.side |= bit((distance_abs < l.side_block_distance) & (lane_abs < l.side_block_lane) & (lane_abs != 0.0));
        masks.lane_outside |= bit(lane > 0.0);
        masks.lane_inside |= bit(0.0 > lane);
        masks.side_near |= bit(l.side_block_distance > distance_abs);
        masks.seen |= bit((0.0 <= distance) & (distance <= l.vision));
    }
    masks
}

/// As `scan_lanes`, eight horses at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn scan_lanes_avx2(view: &ScanView, table: &mut AroundTable, i: usize, n: usize, limits: &ScanLimits) -> ScanMasks {
    use std::arch::x86_64::*;
    let mut masks = ScanMasks::default();
    let l = limits;
    let (my_distance, my_lane, my_speed) =
        (_mm256_set1_ps(view.distance.0[i]), _mm256_set1_ps(view.lane_distance.0[i]), _mm256_set1_ps(view.last_speed.0[i]));
    let zero = _mm256_setzero_ps();
    let sign = _mm256_set1_ps(-0.0);
    let set = _mm256_set1_ps;
    let (surround_side_distance, surround_side_lane) = (set(l.surround_side_distance), set(l.surround_side_lane));
    let (surround_distance, surround_lane, surround_behind) = (set(l.surround_distance), set(l.surround_lane), set(-l.surround_distance));
    let (near_distance, near_lane) = (set(l.near_distance), set(l.near_lane));
    let (compete_fight_lane, compete_fight_distance) = (set(l.compete_fight_lane), set(l.compete_fight_distance));
    let (side_block, side_block_back, side_block_lane) = (set(l.side_block_distance), set(-l.side_block_distance), set(l.side_block_lane));
    let (front_block, vision) = (set(l.front_block_distance), set(l.vision));
    let mut at = 0;
    while at < n {
        // SAFETY: `at + LANES` stays within the columns, because `n` <= MAX_HORSES, a multiple of LANES. Column alignment
        // is that of `HorseColumn`.
        let load = |column: &HorseColumn| unsafe { _mm256_load_ps(column.0.as_ptr().add(at)) };
        let distance = _mm256_sub_ps(load(&view.distance), my_distance);
        let lane = _mm256_sub_ps(load(&view.lane_distance), my_lane);
        let speed = _mm256_sub_ps(my_speed, load(&view.last_speed));
        // `if 0 > x { -x } else { x }`, which keeps -0.0
        let abs = |x: __m256| _mm256_blendv_ps(x, _mm256_xor_ps(x, sign), _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero));
        let (distance_abs, lane_abs, speed_abs) = (abs(distance), abs(lane), abs(speed));
        // SAFETY: as for `load`.
        let put = |column: &mut HorseColumn, x: __m256| unsafe { _mm256_store_ps(column.0.as_mut_ptr().add(at), x) };
        put(&mut table.distance_gap, distance);
        put(&mut table.distance_gap_abs, distance_abs);
        put(&mut table.lane_gap, lane);
        put(&mut table.lane_gap_abs, lane_abs);
        put(&mut table.speed_gap, speed);
        put(&mut table.speed_gap_abs, speed_abs);
        let bits = |x: __m256| (_mm256_movemask_ps(x) as u32) << at;
        let and = |a: __m256, b: __m256| _mm256_and_ps(a, b);
        let ge = |a: __m256, b: __m256| _mm256_cmp_ps::<_CMP_GE_OQ>(a, b);
        let gt = |a: __m256, b: __m256| _mm256_cmp_ps::<_CMP_GT_OQ>(a, b);
        let lt = |a: __m256, b: __m256| _mm256_cmp_ps::<_CMP_LT_OQ>(a, b);
        let le = |a: __m256, b: __m256| _mm256_cmp_ps::<_CMP_LE_OQ>(a, b);
        masks.surround_out |= bits(and(and(ge(surround_side_distance, distance_abs), ge(lane, zero)), ge(surround_side_lane, lane)));
        masks.surround_ahead |= bits(and(and(gt(distance, zero), ge(surround_distance, distance)), ge(surround_lane, lane_abs)));
        masks.surround_behind |= bits(and(and(gt(zero, distance), ge(distance, surround_behind)), ge(surround_lane, lane_abs)));
        masks.near |= !bits(gt(distance_abs, near_distance)) & !bits(gt(lane_abs, near_lane)) & (0xFFu32 << at);
        masks.compete_fight_lane |= bits(ge(compete_fight_lane, lane_abs));
        masks.compete_fight_far |= bits(lt(compete_fight_distance, distance_abs));
        masks.back |= bits(gt(side_block_back, distance));
        masks.front |= bits(and(gt(distance, zero), lt(distance, front_block)));
        masks.side |= bits(and(and(lt(distance_abs, side_block), lt(lane_abs, side_block_lane)), _mm256_cmp_ps::<_CMP_NEQ_OQ>(lane_abs, zero)));
        masks.lane_outside |= bits(gt(lane, zero));
        masks.lane_inside |= bits(gt(zero, lane));
        masks.side_near |= bits(gt(side_block, distance_abs));
        masks.seen |= bits(and(le(zero, distance), le(distance, vision)));
        at += LANES;
    }
    masks
}

/// Scans every other horse: gaps, near / visible lists, front and side blockers, lane space,
/// compete-fight candidates and the surround flags. Runs before any horse moves, so it sees start-of-frame positions
/// (read from `field.scan_view`, see [`snapshot_scan_view`]). The comparisons run on every horse at once, and only the
/// horses that they select go through the rest.
pub fn update_around_horses_param(race: &mut Race, i: usize, dt: f32) {
    let data = race.data;
    let p = &data.param;
    let n = race.horses.len();
    let view = &race.field.scan_view;
    let h = &mut race.horses[i];
    let limits = ScanLimits {
        surround_side_distance: p.Surrounded.DistanceGapAbs1,
        surround_side_lane: p.Surrounded.LaneGap1,
        surround_distance: p.Surrounded.DistanceGap2,
        surround_lane: p.Surrounded.LaneGapAbs2,
        near_distance: p.Near.DistanceThreshold,
        near_lane: p.Near.LaneDistanceThreshold,
        compete_fight_lane: p.CompeteFight.LaneGap,
        compete_fight_distance: p.CompeteFight.DistanceGap,
        side_block_distance: p.Block.SideBlockDistanceGap,
        side_block_lane: p.Block.SideBlockLaneGap,
        front_block_distance: p.Block.FrontBlockDistanceGap,
        vision: visible_distance(h),
    };
    let vision = limits.vision;
    let my_lane = h.lane_distance;
    let lane_max = h.lane_distance_max;
    let self_straight_last = is_straight_last(h);
    let table = &mut *h.ai.around;
    #[cfg(target_arch = "x86_64")]
    let masks = if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2.
        unsafe { scan_lanes_avx2(view, table, i, n, &limits) }
    } else {
        scan_lanes(view, table, i, n, &limits)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let masks = scan_lanes(view, table, i, n, &limits);
    let others = table.others;
    table.back = masks.back & others;
    // The near / visible lists are not cleared: only their first `count` entries are ever read.
    let mut near = masks.near & others;
    let near_count = near.count_ones() as i32;
    let mut at = 0;
    while near != 0 {
        h.ai.near_horses[at] = near.trailing_zeros() as usize;
        at += 1;
        near &= near - 1;
    }
    // Compete-fight candidates: close, side by side, and either horse on the final straight.
    let straight = if self_straight_last { others } else { view.is_straight_last & others };
    let candidates = masks.compete_fight_lane & !masks.compete_fight_far & straight & others;
    let mut leaving = others & !candidates;
    while leaving != 0 {
        let j = leaving.trailing_zeros() as usize;
        if let Some(pos) = h.compete_fight_near.iter().position(|&x| x == j) {
            h.compete_fight_near.remove(pos);
        }
        table.compete_fight_candidate_continue_time.0[j] = 0.0;
        leaving &= leaving - 1;
    }
    let mut entering = candidates;
    while entering != 0 {
        let j = entering.trailing_zeros() as usize;
        if h.compete_fight_near.contains(&j) {
            table.compete_fight_candidate_continue_time.0[j] = dt + table.compete_fight_candidate_continue_time.0[j];
        } else {
            h.compete_fight_near.push(j);
        }
        entering &= entering - 1;
    }
    let ahead = others & !masks.back;
    let mut min_front = f32::MAX;
    let mut block_horse = None;
    let mut front = masks.front & ahead;
    while front != 0 {
        let j = front.trailing_zeros() as usize;
        let distance = table.distance_gap.0[j];
        if check_front_block(p, distance, table.lane_gap_abs.0[j]) && min_front > distance {
            min_front = distance;
            block_horse = Some(j);
        }
        front &= front - 1;
    }
    let side = masks.side & ahead;
    let last = |set: u32| (set != 0).then(|| 31 - set.leading_zeros() as usize);
    h.ai.block_horse_out = last(side & masks.lane_outside);
    h.ai.block_horse_in = last(side & !masks.lane_outside);
    h.ai.block_horse = block_horse;
    let mut in_lane_space = my_lane;
    let mut near_inside = masks.side_near & ahead & masks.lane_inside;
    while near_inside != 0 {
        let gap = table.lane_gap_abs.0[near_inside.trailing_zeros() as usize];
        if in_lane_space > gap {
            in_lane_space = gap;
        }
        near_inside &= near_inside - 1;
    }
    let mut out_lane_space = lane_max - my_lane;
    let mut near_outside = masks.side_near & ahead & masks.lane_outside;
    while near_outside != 0 {
        let gap = table.lane_gap_abs.0[near_outside.trailing_zeros() as usize];
        if out_lane_space > gap {
            out_lane_space = gap;
        }
        near_outside &= near_outside - 1;
    }
    let mut visible_count = 0;
    if vision > 0.0 {
        let mut seen = masks.seen & ahead;
        while seen != 0 {
            let j = seen.trailing_zeros() as usize;
            if is_visible(vision, table.distance_gap.0[j], table.lane_gap_abs.0[j]) {
                h.ai.visibility.visible_horses[visible_count] = j;
                visible_count += 1;
            }
            seen &= seen - 1;
        }
    }
    h.ai.is_around_horse_initialized = true;
    (h.ai.in_lane_space, h.ai.out_lane_space) = (in_lane_space, out_lane_space);
    (h.ai.near_horse_count, h.ai.visibility.visible_horse_count) = (near_count, visible_count as i32);
    h.ai.is_surround_out = masks.surround_out & others != 0;
    h.ai.is_surround_ahead = masks.surround_ahead & others != 0;
    h.ai.is_surround_behind = masks.surround_behind & others != 0;
}

/// `|x|`, but -0.0 stays -0.0 (unlike `f32::abs`), as in the game.
#[inline]
fn abs_keep_neg_zero(x: f32) -> f32 {
    if 0.0 > x {
        -x
    } else {
        x
    }
}

/// Whether a horse `distance_gap` ahead blocks the front. The lane tolerance narrows with distance.
pub fn check_front_block(p: &ParamDefine, distance_gap: f32, lane_gap_abs: f32) -> bool {
    let b = &p.Block;
    if 0.0 >= distance_gap || distance_gap >= b.FrontBlockDistanceGap {
        return false;
    }
    // Evaluated in f64 as one expression.
    let tol = ((1.0 - (distance_gap as f64 / b.FrontBlockDistanceGap as f64) * 0.6f32 as f64) * b.FrontBlockLaneGap as f64) as f32;
    !(lane_gap_abs > tol)
}

/// Whether a horse is inside the vision cone of depth `v`: half-width 1/18 lane at the nose, widening to 0.375 at `v`.
pub fn is_visible(v: f32, distance_gap: f32, lane_gap_abs: f32) -> bool {
    if 0.0 >= v || 0.0 > distance_gap || distance_gap > v {
        return false;
    }
    // Evaluated in f64 as one expression.
    let w = ((distance_gap as f64 / v as f64) * 0.638888896f32 as f64 + 0.111111112f32 as f64) * 0.5;
    !(lane_gap_abs > w as f32)
}

/// Front / side block timers, from the start-of-frame scan.
pub fn update_block_time(race: &mut Race, i: usize, dt: f32) {
    let front = block_horse_front(&race.horses[i]);
    let front_hp_empty = front.is_some_and(|j| race.horses[j].hp <= 0.0);
    let h = &mut race.horses[i];
    let side = h.ai.block_horse_in.is_some() || h.ai.block_horse_out.is_some();
    if front.is_some() {
        h.block_front_accumulate_time += dt;
    }
    h.block_front_continue_time = if front.is_some() { h.block_front_continue_time + dt } else { 0.0 };
    h.block_front_hp_empty_continue_time = if front_hp_empty { h.block_front_hp_empty_continue_time + dt } else { 0.0 };
    if side {
        h.block_side_continue_time += dt;
        // Per-phase maximum; Finished has no entry.
        if let Some(stored) = h.block_side_max_continue_time_by_phase.get_mut(h.phase as usize) {
            if *stored < h.block_side_continue_time {
                *stored = h.block_side_continue_time;
            }
        }
    } else {
        h.block_side_continue_time = 0.0;
    }
}

/// How long the horse one place ahead has stayed near (and near in lane), while the order is unchanged.
pub fn update_in_front_horse_near_time(race: &mut Race, i: usize, dt: f32) {
    let p = &race.data.param.Skill;
    let (cur, prev) = (race.horses[i].cur_order, race.horses[i].prev_order);
    if cur == 0 || cur != prev {
        let h = &mut race.horses[i];
        h.in_front_horse_near_continue_time = 0.0;
        h.in_front_horse_near_lane_continue_time = 0.0;
        return;
    }
    let f = &race.horses[horse_by_order(race, cur - 1)];
    let (f_dist, f_lane) = (f.distance, f.lane_distance);
    let (near_d, near_l) = (p.InfrontHorseNearDistance, p.InfrontHorseNearLaneDistance);
    let h = &mut race.horses[i];
    if f_dist - h.distance > near_d {
        h.in_front_horse_near_continue_time = 0.0;
        h.in_front_horse_near_lane_continue_time = 0.0;
    } else {
        h.in_front_horse_near_continue_time += dt;
        if (h.lane_distance - f_lane).abs() <= near_l {
            h.in_front_horse_near_lane_continue_time += dt;
        } else {
            h.in_front_horse_near_lane_continue_time = 0.0;
        }
    }
}

/// How long the horse one place behind has stayed near (and near in lane), while the order is unchanged.
pub fn update_behind_horse_near_time(race: &mut Race, i: usize, dt: f32) {
    let p = &race.data.param.Skill;
    let (cur, prev) = (race.horses[i].cur_order, race.horses[i].prev_order);
    if cur != race.field.last_order && cur == prev {
        let b = &race.horses[horse_by_order(race, cur + 1)];
        let (b_dist, b_lane) = (b.distance, b.lane_distance);
        let (near_d, near_l) = (p.BehindHorseNearDistance, p.BehindHorseNearLaneDistance);
        let h = &mut race.horses[i];
        if h.distance - b_dist <= near_d {
            h.behind_horse_near_continue_time += dt;
            if (h.lane_distance - b_lane).abs() <= near_l {
                h.behind_horse_near_lane_continue_time += dt;
            } else {
                h.behind_horse_near_lane_continue_time = 0.0;
            }
            return;
        }
    }
    let h = &mut race.horses[i];
    h.behind_horse_near_continue_time = 0.0;
    h.behind_horse_near_lane_continue_time = 0.0;
}

/// As `update_behind_horse_near_time`, once per (distance, lane) threshold pair of the skill params.
pub fn update_behind_horse_near_time_param_set(race: &mut Race, i: usize, dt: f32) {
    let params = &race.data.param.Skill.BehindNearParamArray;
    let (cur, prev) = (race.horses[i].cur_order, race.horses[i].prev_order);
    if cur != race.field.last_order && cur == prev {
        let b = &race.horses[horse_by_order(race, cur + 1)];
        let (b_dist, b_lane) = (b.distance, b.lane_distance);
        let h = &mut race.horses[i];
        let d_gap = h.distance - b_dist;
        let l_gap = (h.lane_distance - b_lane).abs();
        for (t, p) in h.behind_near_lane_times.iter_mut().zip(params) {
            if p.Distance < d_gap || p.LaneDistance < l_gap {
                *t = 0.0;
            } else {
                *t += dt;
            }
        }
    } else {
        race.horses[i].behind_near_lane_times.iter_mut().for_each(|t| *t = 0.0);
    }
}

/// The front-block horse index.
pub fn block_horse_front(h: &Horse) -> Option<usize> {
    h.ai.block_horse
}

/// On the final straight.
pub fn is_straight_last(h: &Horse) -> bool {
    h.is_straight && h.is_final_corner
}

/// Vision depth after skills, at least 0 (NaN → 0).
pub fn visible_distance(h: &Horse) -> f32 {
    max(h.skill_modifier_receiver.apply_modifier(SkillModifierParam::VisibleDistance, h.base_visible_distance), 0.0)
}

/// Free to move inward: no inside blocker and not at the inner limit.
pub fn is_enable_in_move(h: &Horse) -> bool {
    h.ai.block_horse_in.is_none() && !(h.lane_distance_min + 0.0001 >= h.lane_distance)
}

/// Free to move outward: no outside blocker and not at the outer limit.
pub fn is_enable_out_move(h: &Horse) -> bool {
    h.ai.block_horse_out.is_none() && !(h.lane_distance >= h.lane_distance_max)
}

/// Whether there is room for a lane change of `lane_gap_range` (negative = inward).
pub fn check_lane_space(h: &Horse, lane_gap_range: f32) -> bool {
    if 0.0 > lane_gap_range {
        h.ai.in_lane_space >= -lane_gap_range
    } else {
        h.ai.out_lane_space >= lane_gap_range
    }
}

pub fn is_block_front(h: &Horse) -> bool {
    h.ai.block_horse.is_some()
}

pub fn is_block_side(h: &Horse) -> bool {
    h.ai.block_horse_in.is_some() || h.ai.block_horse_out.is_some()
}

/// Blocked in front and unable to move to either side.
pub fn is_block_all(h: &Horse) -> bool {
    is_block_front(h) && !is_enable_in_move(h) && !is_enable_out_move(h)
}

/// All three surround flags set.
pub fn is_surrounded(h: &Horse) -> bool {
    h.ai.is_surround_out && h.ai.is_surround_ahead && h.ai.is_surround_behind
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::test_util::ai_race;

    fn place(race: &mut Race, i: usize, dist: f32, lane: f32) {
        race.horses[i].distance = dist;
        race.horses[i].lane_distance = lane;
    }

    #[test]
    fn constants_bits() {
        assert_eq!(0.638888896f32.to_bits(), 0x3f238e39);
        assert_eq!(0.111111112f32.to_bits(), 0x3de38e39);
        assert_eq!(0.0001f32.to_bits(), 0x38d1b717);
        assert_eq!(0.6f32.to_bits(), 0x3f19999a);
    }

    #[test]
    fn scan_blocks_spaces_vision() {
        let mut race = ai_race(6);
        place(&mut race, 0, 100.0, 0.30);
        place(&mut race, 1, 101.0, 0.30); // front, 1 m: front block
        place(&mut race, 2, 101.0, 0.31); // front, same gap: loses the tie to horse 1 (strict <)
        place(&mut race, 3, 100.5, 0.25); // inside, 0.5 m: side block in, in-space 0.05
        place(&mut race, 4, 99.5, 0.20); // inside, -0.5 m: later side-block-in match wins; space stays 0.05
        place(&mut race, 5, 98.0, 0.30); // behind beyond 1.05 m: isBack, ignored except near
        snapshot_scan_view(&mut race);
        update_around_horses_param(&mut race, 0, 0.0666);
        let ai = &race.horses[0].ai;
        assert_eq!(block_horse_front(&race.horses[0]), Some(1));
        assert_eq!(ai.block_horse_in, Some(4));
        assert_eq!(ai.block_horse_out, Some(2)); // 1.0 m < 1.05 m, lane +0.01
        assert!((ai.in_lane_space - 0.05).abs() < 1e-6);
        assert!((ai.out_lane_space - 0.01).abs() < 1e-6);
        assert!(ai.around.horse(5).is_back);
        let vis = &ai.visibility.visible_horses[..ai.visibility.visible_horse_count as usize];
        assert_eq!(vis, [1, 2, 3]); // cone half-width at 0.5 m ≈ 0.0635 ≥ 0.05
        assert_eq!(ai.near_horse_count, 5);
        assert!(!is_enable_in_move(&race.horses[0]));
        assert!(!is_enable_out_move(&race.horses[0]));
        assert!(is_block_all(&race.horses[0]));
    }
}
