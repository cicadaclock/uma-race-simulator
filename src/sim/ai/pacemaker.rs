//! Pacemaker selection: the horse that position keep measures everyone's distance against.

use crate::sim::defs::RunningStyle;
use crate::sim::race::Race;

#[derive(Clone, Copy, Debug, Default)]
pub struct Pacemaker {
    pub horse: Option<usize>,
    /// From the July 2026 update on ([`update`]): the leader's section at the last re-pick, the section pick, and the
    /// number of re-picks since the last handover that saw an overall leader not of the most-forward style.
    pub section: i32,
    pub section_pick: Option<usize>,
    pub not_most_forward_count: i32,
    /// Whether a horse holds the role through a handover, rather than as the section pick.
    pub overriding: bool,
}

impl Pacemaker {
    /// The pacemaker; horse 0 only before the first `update`.
    pub fn current(&mut self) -> usize {
        *self.horse.get_or_insert(0)
    }
}

/// Before the 2026-07 update: the top-order horse of the most-forward running style (as cached at the end of the
/// previous horse update), re-picked every frame.
///
/// From the 2026-07 update on:
/// - The pick is re-taken only when the overall leader (by last frame's order) enters a new section, from the gate
///   frame (section 1) up to `PositionKeepEndSection`; in between the pacemaker is kept.
/// - Each re-pick that sees an overall leader not of the most-forward style counts +1 and keeps the old section pick;
///   the gate-frame pick never counts (the order there is just horse index).
/// - When the count reaches `PacemakerChangeTopHorseNotMostForwardStyleCount` and the role would change hands, the
///   role passes to the leader's successor ([`successor`]) and the count starts over. The role then stays until the
///   count reaches the threshold again. A count that reaches it with no change of hands is kept.
/// - Once a most-forward-style horse leads again, the section pick is re-taken from the current candidate and the
///   count starts over, on the first frame after that which is not a re-pick.
///
/// `PacemakerChangeMostForwardStyleRange` is not used.
pub fn update(race: &mut Race) {
    let most = race.field.most_forward_running_style;
    let Some(cand) = race.field.running_style_top_order_horses[most as usize] else { return };
    if race.data.era.pre_2026_07_courses {
        race.field.pacemaker.horse = Some(cand);
        return;
    }
    let lead = crate::sim::field::horse_at_order(race, 0);
    let not_most = race.field.pacemaker.horse.is_some() && lead.is_some_and(|l| race.horses[l].horse_data.running_style != most);
    let sec = crate::sim::horse::calc_section(race, lead.unwrap_or(cand));
    let heir = lead.map(|l| successor(race, l));
    let kp = &race.data.param.PositionKeep;
    let threshold = kp.PacemakerChangeTopHorseNotMostForwardStyleCount;
    let end_section = kp.PositionKeepEndSection;
    let pmc = &mut race.field.pacemaker;
    let repick = pmc.horse.is_none() || sec > pmc.section && sec <= end_section;
    if pmc.overriding && !not_most && !repick {
        pmc.overriding = false;
        pmc.not_most_forward_count = 0;
        pmc.section_pick = Some(cand);
        pmc.horse = Some(cand);
    }
    if repick {
        pmc.section = sec;
        if not_most {
            pmc.not_most_forward_count += 1;
            if pmc.not_most_forward_count as f32 >= threshold && heir != pmc.horse {
                pmc.not_most_forward_count = 0;
                pmc.overriding = true;
                pmc.horse = heir;
            }
        } else if !pmc.overriding {
            pmc.section_pick = Some(cand);
            pmc.horse = Some(cand);
        }
    }
}

/// The horse that takes the role from a leader of another style. A Sashi or Oikomi leader passes it to the best placed
/// horse of a more forward style within `PacemakerChangeForwardRunningStyleRange`; any other leader keeps it.
fn successor(race: &Race, leader: usize) -> usize {
    let lead = &race.horses[leader];
    let style = lead.horse_data.running_style as i32;
    if style < RunningStyle::Sashi as i32 {
        return leader;
    }
    let range = race.data.param.PositionKeep.PacemakerChangeForwardRunningStyleRange;
    race.horses
        .iter()
        .enumerate()
        .filter(|(_, h)| (h.horse_data.running_style as i32) < style && lead.distance - h.distance <= range)
        .min_by_key(|(_, h)| h.cur_order)
        .map_or(leader, |(i, _)| i)
}
