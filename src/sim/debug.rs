//! Debug tracing for `compare` (no effect on the simulation).

use crate::sim::race::Race;

/// Debug: `COMPARE_STEP=h1,h2@t0..t1` prints per-step state of the listed horses (env read once).
pub fn step(race: &Race) {
    use std::sync::OnceLock;
    static CFG: OnceLock<Option<(Vec<usize>, f32, f32)>> = OnceLock::new();
    let Some((hs, t0, t1)) = CFG.get_or_init(|| crate::arena::outside(|| {
        let v = std::env::var("COMPARE_STEP").ok()?;
        let (h, t) = v.split_once('@').unwrap_or((&v, "0..1000"));
        let (a, b) = t.split_once("..")?;
        Some((h.split(',').filter_map(|x| x.parse().ok()).collect(), a.parse().ok()?, b.parse().ok()?))
    })) else { return };
    let t = race.recorder.cur_time;
    if t < *t0 || t > *t1 {
        return;
    }
    // The first print creates the standard output buffer, which lasts for the process.
    crate::arena::outside(|| {
        for &i in hs {
            let h = &race.horses[i];
            let cf: Vec<String> = h.compete_fight_near.iter().map(|&k| {
                let a = h.ai.around.horse(k);
                format!("{}:{:.3}/{:.4}", a.info, a.compete_fight_candidate_continue_time, a.speed_gap_abs)
            }).collect();
            println!("STEP t={t:.4} h{i} d={} v={} tv={} hp={}/{} lane={} ord={} cf={}{} g={} ct={} mode={:?} near=[{}]",
                h.distance, h.last_speed, h.ai.target_speed, h.hp, h.max_hp, h.lane_distance, h.cur_order,
                h.is_compete_fight as u8, h.compete_fight_count, h.compete_fight_group, h.is_compete_top as u8,
                crate::sim::ai::position_keep::position_keep_mode(h), cf.join(" "));
            println!("     slope={:?}/{} pow={} guts={} lms={} lss={} m5={} min={} cpa={}", h.slope_type, h.slope_per, crate::sim::physics::stats::pow(h), crate::sim::physics::stats::guts(h), h.lane_move_speed, h.last_self_speed, h.skill_modifier_receiver.apply_modifier(crate::sim::defs::SkillModifierParam::CurrentSpeed, 0.0), h.min_speed, h.ai.conserve_power.accel_value);
            println!("     ctr={} sec={} ls={} lsd={} res={} phase={:?} blk={:?}", h.compete_top_remain_time, crate::sim::horse::calc_section(race, i), crate::sim::physics::last_spurt::is_last_spurt(h) as u8, h.ai.last_spurt_start_distance, h.ai.last_spurt_calc_result, h.phase, h.ai.block_horse);
        }
    });
}
