//! Tick-driven path execution independent of client state.

use azalea::{BlockPos, Vec3};

use super::moves::{LavaPolicy, lava_risk, lava_transition_allowed};
use super::world::{BlockKind, WorldView};
use crate::{Path, PathNode};

/// How far past the launch block's centre to walk before jumping a gap.
///
/// A block is one wide, so 0.35 puts the takeoff close to the lip while still
/// leaving room for the position to be sampled a tick late.
const EDGE_MARGIN: f64 = 0.35;

/// How close to a one-block step to be before hopping onto it.
///
/// A jump carries the bot roughly this far, so jumping any earlier lands back
/// on the same level and any later means walking into the wall first.
const STEP_UP_RANGE: f64 = 1.6;

/// How long to spend backing out of a pinch before retrying the path.
const ESCAPE_TICKS: u32 = 18;
/// How many times to back out before giving up and asking for a fresh plan.
/// Alternating sides means two attempts already cover both ways round.
const MAX_ESCAPES: u32 = 2;
/// How far back to aim when backing out of a pinch.
const ESCAPE_BACK: f64 = 3.0;
/// How far to the side, so the retreat opens a different approach.
const ESCAPE_SIDE: f64 = 2.0;

/// Horizontal distance below which a target has no heading to steer by.
const NO_HEADING_EPSILON: f64 = 0.05;

/// How far below the node being steered at counts as having left the path.
///
/// Every move climbs at most one block, so two and a half is already past any
/// legitimate lag behind the plan while leaving room for a step, a slab and
/// the position being sampled a tick late.
const FELL_OFF_PATH: f64 = 2.5;

/// Ticks of being blocked and making no progress before squaring up on the
/// current block. Long enough that ordinary wall-sliding never triggers it.
const WEDGE_TICKS: u32 = 5;
/// How far off its block's centre the body has to be for squaring up to be
/// worth doing. A body is 0.6 wide in a 1.0 block, so 0.15 is already most of
/// the clearance either side.
const WEDGE_OFF_CENTRE: f64 = 0.15;


#[derive(Debug, Clone)]
pub struct FollowerSettings {
    pub node_radius_xz: f64,
    pub node_y_tolerance: f64,
    pub passed_node_radius_xz: f64,
    pub arrival_radius_xz: f64,
    pub arrival_y_tolerance: f64,
    pub close_enough_xz: f64,
    pub close_enough_y: f64,
    pub stall_ticks: u32,
    pub max_follow_ticks: u32,
    pub max_los_skip: usize,
    pub line_sample_spacing: f64,
    pub body_half_width: f64,
    pub jump_height_threshold: f64,
    pub minimum_turn_degrees: f32,
    pub turn_jitter_degrees: u64,
    pub sprint_break_chance_denominator: u64,
    pub sprint_break_min_ticks: u32,
    pub sprint_break_max_ticks: u32,
    pub escape_hop_first_min_ticks: u32,
    pub escape_hop_first_max_ticks: u32,
    pub escape_hop_second_min_ticks: u32,
    pub escape_hop_second_max_ticks: u32,
    pub yaw_wander_speed: f32,
    pub yaw_wander_degrees: f32,
    pub pitch_wander_speed: f32,
    pub pitch_wander_degrees: f32,
    pub turn_ease: f32,
    pub pitch_clamp_degrees: f32,
    pub minimum_lookahead: f64,
    pub lava_policy: LavaPolicy,
}

impl Default for FollowerSettings {
    fn default() -> Self {
        Self {
            node_radius_xz: 1.0,
            node_y_tolerance: 1.6,
            passed_node_radius_xz: 2.5,
            arrival_radius_xz: 1.0,
            arrival_y_tolerance: 1.2,
            close_enough_xz: 2.0,
            close_enough_y: 1.6,
            stall_ticks: 30,
            max_follow_ticks: 2_400,
            max_los_skip: 12,
            line_sample_spacing: 0.4,
            body_half_width: 0.35,
            jump_height_threshold: 0.55,
            minimum_turn_degrees: 34.0,
            turn_jitter_degrees: 12,
            sprint_break_chance_denominator: 400,
            sprint_break_min_ticks: 15,
            sprint_break_max_ticks: 39,
            escape_hop_first_min_ticks: 6,
            escape_hop_first_max_ticks: 10,
            escape_hop_second_min_ticks: 18,
            escape_hop_second_max_ticks: 23,
            yaw_wander_speed: 0.045,
            yaw_wander_degrees: 4.0,
            pitch_wander_speed: 0.03,
            pitch_wander_degrees: 3.0,
            turn_ease: 0.5,
            pitch_clamp_degrees: 12.0,
            minimum_lookahead: 6.0,
            lava_policy: LavaPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FollowerFrame {
    pub position: Vec3,
    pub on_ground: bool,
    pub horizontal_collision: bool,
    /// Stops movement without consuming the stall budget.
    pub paused: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FollowerDirective {
    Paused,
    Move {
        target: Vec3,
        sprint: bool,
        jump: bool,
        yaw_bias: f32,
        pitch_bias: f32,
        max_turn: f32,
    },
    Arrived,
    Stuck {
        at: BlockPos,
    },
    /// Hold position without walking, but keep looking where the path goes.
    ///
    /// Descending a ladder is the case this exists for: in vanilla you let go
    /// of the movement keys and slide down. Any forward input carries the feet
    /// out of the ladder's block, and the block at the feet is the only thing
    /// that decides whether you are on a ladder at all.
    Wait {
        target: Vec3,
        max_turn: f32,
    },
    Unsafe {
        at: BlockPos,
        next: BlockPos,
    },
}

#[derive(Debug, Clone)]
pub struct PathFollower {
    path: Path,
    settings: FollowerSettings,
    idx: usize,
    last_progress_idx: usize,
    stalled: u32,
    /// Ticks left of backing out of a pinch before trying the path again.
    escaping: u32,
    /// How many times we have backed out on this path.
    escapes_used: u32,
    total_ticks: u32,
    sprint_pause: u32,
    escape_hop_a: u32,
    escape_hop_b: u32,
    yaw_phase: f32,
    pitch_phase: f32,
    max_turn: f32,
    rng: Rng,
}

impl PathFollower {
    pub fn new(path: Path, settings: FollowerSettings, seed: u64) -> Self {
        let mut rng = Rng::seeded(seed);
        let turn_jitter = if settings.turn_jitter_degrees == 0 {
            0
        } else {
            rng.next() % settings.turn_jitter_degrees
        };
        let yaw_phase = (rng.next() % 628) as f32 / 100.0;
        let pitch_phase = (rng.next() % 628) as f32 / 100.0;
        let escape_hop_a = random_tick(
            &mut rng,
            settings.escape_hop_first_min_ticks,
            settings.escape_hop_first_max_ticks,
        );
        let escape_hop_b = random_tick(
            &mut rng,
            settings.escape_hop_second_min_ticks,
            settings.escape_hop_second_max_ticks,
        );
        let idx = usize::from(path.nodes.len() >= 2);
        Self {
            path,
            max_turn: settings.minimum_turn_degrees.max(0.0) + turn_jitter as f32,
            settings,
            idx,
            last_progress_idx: idx,
            stalled: 0,
            escaping: 0,
            escapes_used: 0,
            total_ticks: 0,
            sprint_pause: 0,
            escape_hop_a,
            escape_hop_b,
            yaw_phase,
            pitch_phase,
            rng,
        }
    }

    /// Somewhere with room, away from whatever we are wedged against.
    ///
    /// Straight back on the first try and out to one side on the second, so
    /// two attempts cover both ways round an obstacle instead of retreating
    /// down the same line twice.
    fn escape_target(&self, position: Vec3, blocked_target: Vec3) -> Vec3 {
        let dx = blocked_target.x - position.x;
        let dz = blocked_target.z - position.z;
        let length = (dx * dx + dz * dz).sqrt().max(0.001);
        let (ux, uz) = (dx / length, dz / length);
        // Perpendicular in the horizontal plane, flipped on the second try.
        let side = if self.escapes_used >= 2 { -1.0 } else { 1.0 };
        Vec3 {
            x: position.x - ux * ESCAPE_BACK + -uz * side * ESCAPE_SIDE,
            y: position.y,
            z: position.z - uz * ESCAPE_BACK + ux * side * ESCAPE_SIDE,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn current_node_index(&self) -> usize {
        self.idx
    }

    /// Ticks this follower has spent walking.
    pub fn elapsed_ticks(&self) -> u32 {
        self.total_ticks
    }

    /// Continue another follower's walking clock instead of starting a fresh
    /// one.
    ///
    /// Replacing the path of a journey already in progress must not also
    /// replace how long it has been going: `max_follow_ticks` is the only thing
    /// that catches a route being walked forever, and a goal replanned on a
    /// timer would otherwise reset it faster than it could ever be spent.
    pub fn resume_after(&mut self, ticks: u32) {
        self.total_ticks = self.total_ticks.max(ticks);
    }

    pub fn tick(&mut self, world: &dyn WorldView, frame: FollowerFrame) -> FollowerDirective {
        if self.path.nodes.len() < 2 {
            return FollowerDirective::Arrived;
        }
        if frame.paused {
            self.stalled = 0;
            return FollowerDirective::Paused;
        }
        self.total_ticks = self.total_ticks.saturating_add(1);
        if self.total_ticks > self.settings.max_follow_ticks.max(1) {
            return FollowerDirective::Stuck {
                at: BlockPos::from(&frame.position),
            };
        }

        let nodes = &self.path.nodes;
        while self.idx + 1 < nodes.len() {
            let current = node_center(nodes[self.idx].pos);
            let next = node_center(nodes[self.idx + 1].pos);
            let close = dist_xz(frame.position, current) < self.settings.node_radius_xz.max(0.0)
                && (frame.position.y - current.y).abs() < self.settings.node_y_tolerance.max(0.0);
            let past = dist_xz(frame.position, current)
                < self.settings.passed_node_radius_xz.max(0.0)
                && passed(frame.position, current, next);
            if close || past {
                self.idx += 1;
            } else {
                break;
            }
        }

        let final_target = node_center(nodes[nodes.len() - 1].pos);
        let dxz = dist_xz(frame.position, final_target);
        let dy = (frame.position.y - final_target.y).abs();
        if (dxz < self.settings.arrival_radius_xz.max(0.0)
            && dy < self.settings.arrival_y_tolerance.max(0.0))
            || (dxz < self.settings.close_enough_xz.max(0.0)
                && dy < self.settings.close_enough_y.max(0.0))
        {
            return FollowerDirective::Arrived;
        }

        let next = nodes[self.idx].pos;
        // Off the path entirely, on the floor somewhere under it.
        //
        // No move in the set climbs more than a block at a time, so standing
        // several below the node we are steering at means the route is gone,
        // not that we are behind on it. Missing a jump on a parkour course puts
        // the bot on the ground nine blocks below the next platform, and
        // without this it kept following the path from down there: aiming at a
        // target in the sky, jumping at it on the spot, never moving far enough
        // to trip the stall detector, never replanning. It looks exactly like a
        // bot having a fit, and it is really a bot obeying an itinerary that
        // stopped applying the moment it fell.
        if frame.on_ground && f64::from(next.y) - frame.position.y > FELL_OFF_PATH {
            return FollowerDirective::Stuck {
                at: BlockPos::from(&frame.position),
            };
        }
        if !live_next_node_is_safe(
            world,
            frame.position,
            next,
            self.settings.lava_policy,
            self.settings.line_sample_spacing,
        ) {
            return FollowerDirective::Unsafe {
                at: BlockPos::from(&frame.position),
                next,
            };
        }
        let parkour_blocks = match nodes[self.idx].reached_by {
            crate::MoveKind::Parkour { blocks, .. } => Some(blocks),
            _ => None,
        };
        // Path smoothing normally steers at the furthest node still in sight,
        // which cuts corners nicely on the ground and is exactly wrong in the
        // air: aiming past the landing block flies the bot straight over it.
        // A jump aims at what it is trying to land on, nothing further.
        // A climb is aimed at exactly one rung. Smoothing past it steers the
        // body out of the ladder's own column, and leaving the column is what
        // ends the climb: `OnClimbable` is decided by the block at the feet,
        // not by what the bot is facing.
        let climbing = nodes[self.idx].reached_by == crate::MoveKind::Climb;
        // Fractional-height terrain - snow layers, and any partial block - is
        // where the client and the server disagree about which column a grazing
        // corner of the 0.6-wide body rests on. Corner-smoothing aims across a
        // turn at a further node, and steering diagonally is exactly what pushes
        // the hitbox off its lane and out over a taller neighbour: the +0.25
        // step-up the anticheat cannot reproduce and every measured Simulation
        // flag on this map landed on. The block itself is standable and the
        // physics port is faithful, so the fix is not to change either - it is
        // to keep the body out of the sub-pixel boundary case. On this terrain,
        // aim at the immediate node's centre so the body tracks the column line
        // and never overhangs the block beside it. The planner already prefers
        // the even lane (`grazing_step_penalty`); this keeps the body on it when
        // a crossing is unavoidable.
        let on_fractional = matches!(
            world.block(BlockPos::from(&frame.position)),
            BlockKind::Step(_)
        ) || matches!(world.block(nodes[self.idx].pos), BlockKind::Step(_));
        let target = if parkour_blocks.is_some() || climbing || on_fractional {
            node_center(nodes[self.idx].pos)
        } else {
            let steer_idx =
                furthest_visible(world, frame.position, nodes, self.idx, &self.settings);
            node_center(nodes[steer_idx].pos)
        };

        // Feet in water: there is no ground under them, so `on_ground` stays
        // false and every jump condition below is dead. In vanilla the jump key
        // is the swim-up input, and holding it is both what keeps the bot at the
        // surface while crossing and the only way onto a bank; letting go is how
        // vanilla sinks, so it is released only when the plan goes downward.
        let feet = BlockPos::from(&frame.position);
        let in_water = world.block(feet) == BlockKind::Water;
        let swimming_up = in_water && target.y.floor() as i32 >= feet.y;
        // On a ladder the jump key is the climb input and `on_ground` is false,
        // exactly as in water. Holding it is what ascends; releasing it is how
        // vanilla slides back down, so it is released as soon as the plan stops
        // going up. The check is on the block under the feet rather than on the
        // move kind, because the rung the bot is standing on is the authority
        // on whether it is on a ladder at all.
        let on_ladder = world.block(feet) == BlockKind::Climbable;
        let climbing_up = on_ladder && target.y > frame.position.y + 0.1;

        // Backing out of a pinch. Replanning on the spot is useless: the
        // planner is deterministic, so from the same block it returns the same
        // path and the bot wedges against the same corner until it runs out of
        // legs. Moving somewhere with room first is what makes the next plan
        // come out different.
        if self.escaping > 0 {
            self.escaping -= 1;
            return FollowerDirective::Move {
                target: self.escape_target(frame.position, target),
                sprint: false,
                jump: swimming_up
                    || climbing_up
                    || self.escaping.is_multiple_of(6) && frame.on_ground,
                yaw_bias: 0.0,
                pitch_bias: 0.0,
                max_turn: self.max_turn,
            };
        }

        if self.idx > self.last_progress_idx {
            self.last_progress_idx = self.idx;
            self.stalled = 0;
            self.escapes_used = 0;
        } else {
            self.stalled = self.stalled.saturating_add(1);
            if self.stalled > self.settings.stall_ticks.max(1) {
                if self.escapes_used < MAX_ESCAPES {
                    self.escapes_used += 1;
                    self.escaping = ESCAPE_TICKS;
                    self.stalled = 0;
                    if std::env::var("PF_NAV_DEBUG").is_ok() {
                        eprintln!(
                            "nav escape {} at {:.1},{:.1},{:.1} (node {} of {})",
                            self.escapes_used,
                            frame.position.x,
                            frame.position.y,
                            frame.position.z,
                            self.idx,
                            nodes.len(),
                        );
                    }
                    return FollowerDirective::Move {
                        target: self.escape_target(frame.position, target),
                        sprint: false,
                        jump: false,
                        yaw_bias: 0.0,
                        pitch_bias: 0.0,
                        max_turn: self.max_turn,
                    };
                }
                return FollowerDirective::Stuck {
                    at: BlockPos::from(&frame.position),
                };
            }
        }

        // A gap jump has nothing to walk into, so waiting for a horizontal
        // collision means walking off the edge instead of jumping. Being on
        // the ground is the whole condition: the bot is only on the ground
        // while it is still on the launch block. Gating on distance to that
        // block's centre missed the window entirely, because the path index
        // only advances once we are already past the centre, so the bot walked
        // off the edge without ever jumping.
        // Jump at the edge, not on arrival at the block. The whole arc is
        // measured from where the feet leave the ground, so taking off from
        // the middle of the launch block wastes half a block of it and lands
        // short. Waiting until we have advanced most of the way across the
        // block puts the takeoff at the lip, which is where a player jumps.
        // Wedged on a corner: line up on our own block before pushing on.
        //
        // A body is 0.6 wide and a doorway is 1.0, so the gap either side is
        // 0.2 and the only way through is near the middle. Steering straight at
        // the next block aims the whole push along the blocked direction, and
        // the sideways component that would clear the corner shrinks as the bot
        // turns to face the target - so it presses into the edge, slides a
        // fraction, and stops.
        //
        // Measured at the foot of a ladder beside a tree: the bot sat with
        // `z` frozen to four decimal places for a hundred ticks, overlapping
        // the trunk by four hundredths of a block, jumping on the spot because
        // being blocked is also what triggers a hop. Four hundredths.
        //
        // Aiming at the middle of the block it is already standing in backs it
        // off the corner and squares it up, and then the next attempt goes
        // straight through. This is what a person does after bumping a doorway,
        // and it waits for the stall counter so that ordinary wall-sliding -
        // which is a collision every tick while making perfectly good progress
        // - is left alone.
        let target = if frame.horizontal_collision && self.stalled >= WEDGE_TICKS {
            let here = node_center(BlockPos::from(&frame.position));
            if dist_xz(frame.position, here) > WEDGE_OFF_CENTRE {
                Vec3 {
                    x: here.x,
                    y: target.y,
                    z: here.z,
                }
            } else {
                target
            }
        } else {
            target
        };

        let launch = node_center(nodes[self.idx.saturating_sub(1)].pos);
        let planned = dist_xz(launch, target);
        let remaining = dist_xz(frame.position, target);
        let at_edge = remaining <= planned - EDGE_MARGIN;
        let taking_off = parkour_blocks.is_some() && at_edge;
        if let Some(blocks) = parkour_blocks
            && std::env::var("PF_PARKOUR_DEBUG").is_ok()
        {
            eprintln!(
                "parkour follow: at {:.2},{:.2},{:.2} on_ground={} -> target {:.2},{:.2},{:.2} dist {:.2} blocks={blocks}",
                frame.position.x,
                frame.position.y,
                frame.position.z,
                frame.on_ground,
                target.x,
                target.y,
                target.z,
                dist_xz(frame.position, target),
            );
        }

        // Climbing terrain is planned as a chain of one-block step-ups, but
        // the only trigger used to be "we bumped into something", so the bot
        // walked into every step, stalled, and only then hopped. On a slope
        // that reads as a bot that cannot climb. If the path says the next
        // node is a step up, jump when we are close enough to land on it.
        // Measured against the node we are about to arrive at, never against the
        // smoothed steering target.
        //
        // Smoothing aims at the furthest node still in sight, up to twelve
        // ahead, and it is allowed to differ in height by a block. So "is there
        // a step in front of me" was being answered by a block up to twelve
        // paces away: on level ground beside a wall, with the route rising later
        // on, the bot read a node it had not reached yet as a step under its
        // feet and hopped, on every tick it was in contact with the wall. Over a
        // village-to-mine round trip that was the single largest source of
        // jumps, and 91% of all jumps in the worst run gained no height at all.
        //
        // The block being stepped onto is `nodes[idx]`, and it is the only
        // honest answer to both questions below: how far away the step is, and
        // whether it is above us.
        let immediate = node_center(nodes[self.idx].pos);
        let stepping_up = matches!(nodes[self.idx].reached_by, crate::MoveKind::Jump)
            && dist_xz(frame.position, immediate) < STEP_UP_RANGE
            && immediate.y - frame.position.y > self.settings.jump_height_threshold.max(0.0);

        // Sliding down a ladder is done by letting go, not by walking. The
        // rung below is directly underfoot, so any forward input is sideways
        // motion that leaves the shaft: the bot fell onto the floor beside the
        // ladder, replanned, climbed back up, and repeated until it ran out of
        // legs. Six consecutive legs died on one ladder in the hub village.
        if on_ladder && target.y < frame.position.y - 0.1 {
            return FollowerDirective::Wait {
                target,
                max_turn: self.max_turn,
            };
        }

        // Going up a ladder means pressing into the wall it hangs on.
        //
        // The rung above is directly overhead, so steering at it gives almost
        // no horizontal direction, and with nothing to steer by the bot simply
        // keeps the heading it arrived on - which points across the shaft. It
        // climbed at exactly the right speed while walking out of its own
        // column, left the ladder a block and a half up, and fell. Descending
        // already has this problem and solves it by refusing to walk at all;
        // going up cannot do that, because the climb needs the forward input.
        //
        // Aiming at the block the ladder is fixed to holds the body against it
        // for the whole climb, which is both what keeps the feet in the ladder
        // block and what a player's hand is doing on the keyboard.
        let target = if climbing_up {
            ladder_wall(world, feet).map_or(target, |wall| Vec3 {
                x: f64::from(wall.x) + 0.5,
                y: target.y,
                z: f64::from(wall.z) + 0.5,
            })
        } else {
            target
        };

        // Blocked, with the block we are trying to enter above us: hop onto it.
        // Same correction as `stepping_up` above, and for the same reason - a
        // wall beside the path is not a step, however high the route goes later.
        let blocked_below_a_rise = frame.horizontal_collision
            && immediate.y - frame.position.y > self.settings.jump_height_threshold.max(0.0);
        let shove = frame.horizontal_collision
            && (self.stalled == self.escape_hop_a || self.stalled == self.escape_hop_b);
        let jump = swimming_up
            || climbing_up
            || frame.on_ground && (taking_off || stepping_up || blocked_below_a_rise || shove);
        if jump && jump_debug() {
            let reason = if swimming_up {
                "swim"
            } else if climbing_up {
                "climb"
            } else if taking_off {
                "parkour"
            } else if stepping_up {
                "step-up"
            } else if blocked_below_a_rise {
                "blocked-below-a-rise"
            } else {
                "shove"
            };
            eprintln!("jump {reason}");
        }
        // Sprint state is part of the jump, not a preference: a sprint jump
        // travels about 4 blocks and a walking jump about 2.5, so sprinting a
        // 2 block hop sails straight over the landing block, and walking a 3
        // block one drops into the gap. Either way the bot falls, so parkour
        // pins sprint in both directions and overrides the idle sprint breaks.
        let sprint = if on_ladder {
            // Sprinting is not a thing the server will predict on a ladder:
            // climb speed is a constant, so asserting sprint only asks for a
            // velocity it will not reproduce.
            false
        } else if in_water {
            // Vanilla only sprints in water in the swimming pose, which needs
            // the body submerged and a different animation. Wading with sprint
            // asserted asks the server for a speed it will not predict, and in
            // shallow water `on_ground` is true, so the ground test below would
            // otherwise hold it down.
            false
        } else if let Some(blocks) = parkour_blocks {
            // Only ever assert sprint from the ground. The plugin sends a
            // sprint-or-walk event every tick, so holding sprint true through
            // the airborne half of a jump keeps flipping a flag the server
            // predicts differently, which Grim reads as a per-tick offset.
            // Jump distance is set by the velocity at takeoff anyway.
            blocks >= 3 && frame.on_ground
        } else if self.sprint_pause > 0 {
            self.sprint_pause -= 1;
            false
        } else if frame.on_ground
            && self
                .rng
                .next()
                .is_multiple_of(self.settings.sprint_break_chance_denominator.max(1))
        {
            self.sprint_pause = random_tick(
                &mut self.rng,
                self.settings.sprint_break_min_ticks,
                self.settings.sprint_break_max_ticks,
            );
            false
        } else {
            frame.on_ground
        };
        let t = self.total_ticks as f32;
        // Do not turn on the tick we leave the ground. A sprint jump's
        // horizontal boost is computed from yaw, so if the yaw is still
        // sliding when the impulse is applied, the client's boost points a
        // fraction differently from the one the server reproduces from the
        // rotation it last received. It shows up as a single small offset at
        // takeoff that then decays over the following ticks. Aiming first and
        // then jumping is also what a player does.
        //
        // Only for an actual takeoff, which means only from the ground. The
        // jump input is also the climb input and the swim-up input, and both of
        // those are *held*, so testing the input alone froze the bot's aim for
        // the whole of every ladder. It arrived at the shaft facing across it,
        // could never turn to face the wall, and walked out of the column while
        // climbing at exactly the right speed - three rungs up, every time.
        // Neither a ladder nor water applies a jump impulse, so neither has the
        // offset this guard exists to prevent.
        let turning = if jump && frame.on_ground {
            0.0
        } else {
            self.max_turn
        };
        FollowerDirective::Move {
            target,
            sprint,
            jump,
            yaw_bias: (t * self.settings.yaw_wander_speed + self.yaw_phase).sin()
                * self.settings.yaw_wander_degrees,
            pitch_bias: (t * self.settings.pitch_wander_speed + self.pitch_phase).sin()
                * self.settings.pitch_wander_degrees,
            max_turn: turning,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn steering_direction(
    position: Vec3,
    current_yaw: f32,
    current_pitch: f32,
    target: Vec3,
    yaw_bias: f32,
    pitch_bias: f32,
    max_turn: f32,
    settings: &FollowerSettings,
) -> (f32, f32) {
    let max_turn = max_turn.max(0.0);
    let dx = target.x - position.x;
    let dz = target.z - position.z;
    // A target directly overhead or underfoot has no direction to face, and
    // `atan2(-0.0, 0.0)` is not "no direction", it is due south. Descending a
    // ladder aims at the rung below, i.e. exactly this case, so the bot span
    // to face south and then walked off the ladder, fell to the floor, climbed
    // back up and did it again. Keep the current heading when there is no
    // horizontal component to steer by.
    let desired_yaw = if dx.abs() < NO_HEADING_EPSILON && dz.abs() < NO_HEADING_EPSILON {
        current_yaw
    } else {
        f64::atan2(-dx, dz).to_degrees() as f32 + yaw_bias
    };
    let eye_y = position.y + 1.62;
    let target_eye_y = target.y + 1.62;
    let horizontal = (dx * dx + dz * dz)
        .sqrt()
        .max(settings.minimum_lookahead.max(0.001));
    let desired_pitch =
        (f64::atan2(eye_y - target_eye_y, horizontal).to_degrees() as f32 + pitch_bias).clamp(
            -settings.pitch_clamp_degrees.abs(),
            settings.pitch_clamp_degrees.abs(),
        );
    let ease = settings.turn_ease.clamp(0.0, 1.0);
    (
        wrap_degrees(
            current_yaw + (angle_delta(current_yaw, desired_yaw) * ease).clamp(-max_turn, max_turn),
        ),
        current_pitch + ((desired_pitch - current_pitch) * ease).clamp(-max_turn, max_turn),
    )
}

/// Fold an angle back into -180..180.
///
/// Turning is expressed as "current heading plus a small delta", and nothing
/// ever brought the sum back into range, so a bot that spent an hour walking
/// clockwise round a mountain reached a yaw of -12187 degrees: thirty-three
/// full rotations, still pointing the right way, still growing. Sine and cosine
/// do not care, but the number is sent to the server as a 32-bit float, and its
/// precision is spent on the revolutions rather than on the heading - by ten
/// thousand degrees the representable steps are coarser than the tenth of a
/// degree the bot is trying to steer with. No real client ever sends this.
fn wrap_degrees(degrees: f32) -> f32 {
    let wrapped = degrees % 360.0;
    if wrapped > 180.0 {
        wrapped - 360.0
    } else if wrapped < -180.0 {
        wrapped + 360.0
    } else {
        wrapped
    }
}

pub fn furthest_visible(
    world: &dyn WorldView,
    position: Vec3,
    nodes: &[PathNode],
    idx: usize,
    settings: &FollowerSettings,
) -> usize {
    if nodes.is_empty() {
        return 0;
    }
    let idx = idx.min(nodes.len() - 1);
    let mut best = idx;
    let y = nodes[idx].pos.y;
    let limit = idx
        .saturating_add(settings.max_los_skip)
        .min(nodes.len() - 1);
    for (j, node) in nodes.iter().enumerate().take(limit + 1).skip(idx + 1) {
        // One block of height difference is still a straight walk: auto-step
        // covers half a block and the follower jumps the rest. Demanding an
        // exactly level run stopped smoothing at the first stair on the route,
        // which on this map is 38% of nodes, and an unsmoothed follower steers
        // at the block one step ahead and weaves along every grid staircase.
        if (node.pos.y - y).abs() > 1
            || !line_walkable(world, position, node_center(node.pos), y, settings)
        {
            break;
        }
        best = j;
    }
    best
}

pub fn line_walkable(
    world: &dyn WorldView,
    from: Vec3,
    to: Vec3,
    feet_y: i32,
    settings: &FollowerSettings,
) -> bool {
    let dx = to.x - from.x;
    let dz = to.z - from.z;
    let distance = (dx * dx + dz * dz).sqrt();
    if distance < 1e-6 {
        return true;
    }
    let perp_x = -dz / distance;
    let perp_z = dx / distance;
    let samples = (distance / settings.line_sample_spacing.max(0.05))
        .ceil()
        .max(1.0) as i32;
    for i in 1..=samples {
        let t = i as f64 / samples as f64;
        let cx = from.x + dx * t;
        let cz = from.z + dz * t;
        // Level, or one step up or down. The path already said this run is
        // walkable; the job here is to reject shortcuts across a hole or a
        // wall, not to insist the floor never changes height.
        let column = BlockPos::new(cx.floor() as i32, feet_y, cz.floor() as i32);
        let Some(center) = [0, 1, -1]
            .into_iter()
            .map(|dy| BlockPos::new(column.x, feet_y + dy, column.z))
            .find(|p| world.standable(*p))
        else {
            return false;
        };
        // `standable` counts open water, because a swimmer holds position
        // there. Smoothing is about walking, though: steering straight at a
        // node across a pond walks the bot into the pond, which is the one
        // place the planner just went out of its way to avoid.
        if matches!(
            world.block(center),
            BlockKind::Water | BlockKind::Climbable
        ) {
            return false;
        }
        if position_lava_risk(world, center, settings.lava_policy) > 0 {
            return false;
        }
        for side in [
            settings.body_half_width.max(0.0),
            -settings.body_half_width.max(0.0),
        ] {
            let x = (cx + perp_x * side).floor() as i32;
            let z = (cz + perp_z * side).floor() as i32;
            // A slab or stair at foot height is something the body walks *onto*,
            // not into: auto-step lifts over it without slowing down. Treating
            // it as a wall here refused to smooth past every slab on the map,
            // and this map's floors are largely slabs.
            if matches!(
                world.block(BlockPos::new(x, center.y, z)),
                BlockKind::Solid | BlockKind::Fence
            ) || matches!(
                world.block(BlockPos::new(x, center.y + 1, z)),
                BlockKind::Solid | BlockKind::Fence
            ) {
                return false;
            }
        }
    }
    true
}

fn live_next_node_is_safe(
    world: &dyn WorldView,
    from: Vec3,
    to: BlockPos,
    policy: LavaPolicy,
    sample_spacing: f64,
) -> bool {
    let from_block = BlockPos::from(&from);
    let from_risk = position_lava_risk(world, from_block, policy);
    let to_risk = position_lava_risk(world, to, policy);
    if !lava_transition_allowed(from_risk, to_risk, policy) {
        return false;
    }
    if from_risk > 0 || matches!(policy, LavaPolicy::Penalized) {
        return true;
    }
    let to = node_center(to);
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let dz = to.z - from.z;
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    let samples = (distance / sample_spacing.max(0.05)).ceil().max(1.0) as i32;
    (1..=samples).all(|i| {
        let t = i as f64 / samples as f64;
        let sample = BlockPos::new(
            (from.x + dx * t).floor() as i32,
            (from.y + dy * t).floor() as i32,
            (from.z + dz * t).floor() as i32,
        );
        position_lava_risk(world, sample, policy) == 0
    })
}

fn position_lava_risk(world: &dyn WorldView, pos: BlockPos, policy: LavaPolicy) -> u32 {
    match policy {
        LavaPolicy::Forbidden { clearance } => lava_risk(pos, world, clearance),
        LavaPolicy::Penalized => 0,
    }
}

pub fn node_center(pos: BlockPos) -> Vec3 {
    Vec3::new(pos.x as f64 + 0.5, pos.y as f64, pos.z as f64 + 0.5)
}

fn dist_xz(a: Vec3, b: Vec3) -> f64 {
    ((a.x - b.x).powi(2) + (a.z - b.z).powi(2)).sqrt()
}

fn passed(pos: Vec3, node: Vec3, next: Vec3) -> bool {
    (pos.x - node.x) * (next.x - node.x) + (pos.z - node.z) * (next.z - node.z) > 0.0
}

/// The block a ladder is fixed to, if the feet are in one.
///
/// A ladder's collision box is a sliver against this block, and its whole
/// purpose here is to give the climb something to press against.
/// `PF_JUMP_DEBUG=1` names the rule behind every jump the follower asks for.
///
/// Worth having permanently: a bot that hops constantly looks like one bug and
/// is usually another, and the five rules that can raise a jump are impossible
/// to tell apart from outside.
fn jump_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PF_JUMP_DEBUG").is_ok())
}

fn ladder_wall(world: &dyn WorldView, feet: BlockPos) -> Option<BlockPos> {
    if world.block(feet) != BlockKind::Climbable {
        return None;
    }
    [(1, 0), (-1, 0), (0, 1), (0, -1)]
        .into_iter()
        .map(|(dx, dz)| BlockPos::new(feet.x + dx, feet.y, feet.z + dz))
        .find(|pos| world.block(*pos) == BlockKind::Solid)
}

fn angle_delta(from: f32, to: f32) -> f32 {
    let mut delta = (to - from) % 360.0;
    if delta > 180.0 {
        delta -= 360.0;
    }
    if delta < -180.0 {
        delta += 360.0;
    }
    delta
}

fn random_tick(rng: &mut Rng, min: u32, max: u32) -> u32 {
    let (min, max) = (min.min(max), min.max(max));
    let span = max.saturating_sub(min).saturating_add(1).max(1);
    min.saturating_add((rng.next() % u64::from(span)) as u32)
}

#[derive(Debug, Clone)]
struct Rng(u64);
impl Rng {
    fn seeded(seed: u64) -> Self {
        Self((seed ^ 0x9E37_79B9_7F4A_7C15).max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Grid(HashMap<(i32, i32, i32), BlockKind>);
    impl WorldView for Grid {
        fn block(&self, pos: BlockPos) -> BlockKind {
            self.0
                .get(&(pos.x, pos.y, pos.z))
                .copied()
                .unwrap_or(BlockKind::Air)
        }
    }
    fn floor() -> Grid {
        Grid((-2..=8).map(|x| ((x, 63, 0), BlockKind::Solid)).collect())
    }
    fn path() -> Path {
        Path {
            nodes: (0..=4)
                .map(|x| PathNode {
                    pos: BlockPos::new(x, 64, 0),
                    reached_by: if x == 0 {
                        crate::MoveKind::Start
                    } else {
                        crate::MoveKind::Walk
                    },
                })
                .collect(),
            total_cost: 40,
        }
    }

    #[test]
    fn follower_arrives_and_pauses_without_stalling() {
        let world = floor();
        let mut follower = PathFollower::new(path(), FollowerSettings::default(), 7);
        assert_eq!(
            follower.tick(
                &world,
                FollowerFrame {
                    position: Vec3::new(0.5, 64.0, 0.5),
                    on_ground: true,
                    horizontal_collision: false,
                    paused: true
                }
            ),
            FollowerDirective::Paused
        );
        assert_eq!(
            follower.tick(
                &world,
                FollowerFrame {
                    position: Vec3::new(4.5, 64.0, 0.5),
                    on_ground: true,
                    horizontal_collision: false,
                    paused: false
                }
            ),
            FollowerDirective::Arrived
        );
    }

    /// Scraping a wall on level ground is not a reason to jump.
    ///
    /// The jump rules used to ask whether the *smoothed steering target* was
    /// above the feet, and smoothing aims up to twelve nodes ahead at a node
    /// allowed to differ in height by a block. So a bot walking a flat stretch
    /// beside a wall, on a route that climbs later, read a node it had not
    /// reached as a step under its feet and hopped on every tick it was in
    /// contact. Measured on a village-to-mine round trip: the largest single
    /// source of jumps, and in the worst run 91% of all jumps gained no height.
    #[test]
    fn a_wall_beside_a_level_path_does_not_make_the_bot_hop() {
        // Flat floor the whole way, with a wall along one side to scrape.
        let mut blocks = HashMap::new();
        for x in -2..=14 {
            blocks.insert((x, 63, 0), BlockKind::Solid);
            for h in 0..=1 {
                blocks.insert((x, 64 + h, 1), BlockKind::Solid);
            }
        }
        // The route is level underfoot but ends a block higher, which is what
        // the smoothed target used to pick up.
        let mut nodes: Vec<PathNode> = (0..=9)
            .map(|x| PathNode {
                pos: BlockPos::new(x, 64, 0),
                reached_by: if x == 0 {
                    crate::MoveKind::Start
                } else {
                    crate::MoveKind::Walk
                },
            })
            .collect();
        nodes.push(PathNode {
            pos: BlockPos::new(10, 65, 0),
            reached_by: crate::MoveKind::Jump,
        });
        blocks.insert((10, 64, 0), BlockKind::Solid);
        let world = Grid(blocks);

        let mut follower = PathFollower::new(
            Path { nodes, total_cost: 100 },
            FollowerSettings::default(),
            11,
        );
        // Standing on the second node, pressed against the wall, far from the
        // step at the far end.
        let directive = follower.tick(
            &world,
            FollowerFrame {
                position: Vec3::new(1.5, 64.0, 0.5),
                on_ground: true,
                horizontal_collision: true,
                paused: false,
            },
        );
        match directive {
            FollowerDirective::Move { jump, .. } => {
                assert!(!jump, "hopped while scraping a wall on level ground");
            }
            other => panic!("expected a Move, got {other:?}"),
        }
    }

    /// But a step directly in front still gets hopped, promptly.
    #[test]
    fn a_step_in_front_is_still_jumped() {
        let mut blocks = HashMap::new();
        for x in -2..=1 {
            blocks.insert((x, 63, 0), BlockKind::Solid);
        }
        // A raised shelf from x=2 onward, which the plan steps up onto. It runs
        // well past the step so the far end is not close enough to count as
        // arriving, which would end the tick before any jump is decided.
        for x in 2..=10 {
            blocks.insert((x, 64, 0), BlockKind::Solid);
        }
        let world = Grid(blocks);

        let mut nodes = vec![
            PathNode { pos: BlockPos::new(0, 64, 0), reached_by: crate::MoveKind::Start },
            PathNode { pos: BlockPos::new(1, 64, 0), reached_by: crate::MoveKind::Walk },
            PathNode { pos: BlockPos::new(2, 65, 0), reached_by: crate::MoveKind::Jump },
        ];
        nodes.extend((3..=10).map(|x| PathNode {
            pos: BlockPos::new(x, 65, 0),
            reached_by: crate::MoveKind::Walk,
        }));
        let mut follower =
            PathFollower::new(Path { nodes, total_cost: 120 }, FollowerSettings::default(), 3);
        // Walk at the step and watch the whole approach: the follower advances
        // its node index as it closes, so which tick carries the hop is an
        // implementation detail. That one of them does is not.
        let mut jumped = false;
        for step in 0..6 {
            let directive = follower.tick(
                &world,
                FollowerFrame {
                    position: Vec3::new(1.2 + 0.15 * f64::from(step), 64.0, 0.5),
                    on_ground: true,
                    horizontal_collision: false,
                    paused: false,
                },
            );
            if let FollowerDirective::Move { jump: true, .. } = directive {
                jumped = true;
            }
        }
        assert!(jumped, "never hopped onto a step directly in front");
    }

    #[test]
    fn fractional_terrain_pins_the_steer_to_the_immediate_node() {
        // A straight one-wide run of eight nodes. Over plain ground the follower
        // smooths its steer far down the line and cuts corners; over
        // fractional-height blocks - snow - that diagonal drift is what lifts a
        // grazing corner of the 0.6-wide body onto a taller neighbour and
        // desyncs it against the anticheat. So on that terrain the steer is
        // pinned to the immediate node and the body stays centred on its lane.
        let straight = || Path {
            nodes: (0..=8)
                .map(|x| PathNode {
                    pos: BlockPos::new(x, 64, 0),
                    reached_by: if x == 0 {
                        crate::MoveKind::Start
                    } else {
                        crate::MoveKind::Walk
                    },
                })
                .collect(),
            total_cost: 80,
        };
        let frame = || FollowerFrame {
            position: Vec3::new(0.5, 64.0, 0.5),
            on_ground: true,
            horizontal_collision: false,
            paused: false,
        };

        // Plain ground below, air to walk through: the steer smooths to the far
        // end of the visible run.
        let smooth = Grid((-2..=10).map(|x| ((x, 63, 0), BlockKind::Solid)).collect());
        let mut follower = PathFollower::new(straight(), FollowerSettings::default(), 7);
        let FollowerDirective::Move { target, .. } = follower.tick(&smooth, frame()) else {
            panic!("expected a Move over plain ground");
        };
        assert!(
            target.x > 2.0,
            "plain ground should smooth the steer far down the line, got x={}",
            target.x
        );

        // Snow underfoot (a partial-height block in the feet's own cell): the
        // same run pins the steer to the immediate node's centre.
        let snowy = Grid((-2..=10).map(|x| ((x, 64, 0), BlockKind::Step(4))).collect());
        let mut follower = PathFollower::new(straight(), FollowerSettings::default(), 7);
        let FollowerDirective::Move { target, .. } = follower.tick(&snowy, frame()) else {
            panic!("expected a Move over snow");
        };
        assert!(
            (target.x - 1.5).abs() < 1e-9,
            "snow should pin the steer to the immediate node centre, got x={}",
            target.x
        );
    }

    #[test]
    fn shortcuts_respect_lava_and_body_clearance() {
        let mut world = floor();
        world.0.insert((2, 63, 1), BlockKind::Lava);
        assert!(!line_walkable(
            &world,
            Vec3::new(0.5, 64.0, 0.5),
            Vec3::new(4.5, 64.0, 0.5),
            64,
            &FollowerSettings::default()
        ));

        // A wall at the body's edge still blocks the shortcut.
        world.0.remove(&(2, 63, 1));
        world.0.insert((2, 64, 1), BlockKind::Solid);
        assert!(!line_walkable(
            &world,
            Vec3::new(0.5, 64.0, 0.8),
            Vec3::new(4.5, 64.0, 0.8),
            64,
            &FollowerSettings::default()
        ));

        // A slab at the body's edge does not. Auto-step lifts the body over it
        // without slowing down, and refusing to smooth past one means refusing
        // to smooth anywhere on a map whose floors are slabs.
        world.0.insert((2, 64, 1), BlockKind::Step(8));
        assert!(line_walkable(
            &world,
            Vec3::new(0.5, 64.0, 0.8),
            Vec3::new(4.5, 64.0, 0.8),
            64,
            &FollowerSettings::default()
        ));
    }

    #[test]
    fn shortcuts_step_up_and_down_but_not_over_a_hole() {
        // A one block rise mid-run is still a straight walk.
        let mut world = floor();
        world.0.insert((2, 64, 0), BlockKind::Solid);
        assert!(line_walkable(
            &world,
            Vec3::new(0.5, 64.0, 0.5),
            Vec3::new(4.5, 64.0, 0.5),
            64,
            &FollowerSettings::default()
        ));

        // A hole in the floor is not.
        let mut world = floor();
        world.0.remove(&(2, 63, 0));
        assert!(!line_walkable(
            &world,
            Vec3::new(0.5, 64.0, 0.5),
            Vec3::new(4.5, 64.0, 0.5),
            64,
            &FollowerSettings::default()
        ));
    }

    #[test]
    fn plants_are_walked_through_not_into() {
        let mut world = floor();
        // Grass where the feet go, on a floor that is otherwise fine.
        world.0.insert((2, 64, 0), BlockKind::Passable);
        assert!(world.standable(BlockPos::new(2, 64, 0)));
        assert!(line_walkable(
            &world,
            Vec3::new(0.5, 64.0, 0.5),
            Vec3::new(4.5, 64.0, 0.5),
            64,
            &FollowerSettings::default()
        ));
    }
    /// Falling off the route has to end the leg, not be followed from below.
    ///
    /// A missed jump on a parkour course leaves the bot on the floor with the
    /// next platform still overhead. It used to keep steering at it: aiming at
    /// a target in the sky and hopping on the spot, going nowhere but moving
    /// enough that the stall detector never fired, so the leg ran until the
    /// follow timeout while the bot appeared to be having a fit.
    #[test]
    fn landing_far_below_the_next_node_ends_the_leg() {
        let mut blocks: HashMap<(i32, i32, i32), BlockKind> = HashMap::new();
        for x in -2..=8 {
            blocks.insert((x, 54, 0), BlockKind::Solid);
        }
        let world = Grid(blocks);

        // A route across platforms at y=64, being followed by a bot that is
        // standing on the floor ten blocks below them.
        let mut follower = PathFollower::new(path(), FollowerSettings::default(), 7);
        let directive = follower.tick(
            &world,
            FollowerFrame {
                position: Vec3::new(0.5, 55.0, 0.5),
                on_ground: true,
                horizontal_collision: false,
                paused: false,
            },
        );
        assert!(
            matches!(directive, FollowerDirective::Stuck { .. }),
            "kept following a path ten blocks overhead: {directive:?}"
        );
    }

    /// The same check must not fire on ordinary climbing.
    ///
    /// Every move gains at most a block, so a bot one step behind a rising path
    /// is normal and has to keep walking.
    #[test]
    fn being_one_step_below_the_next_node_is_normal() {
        let world = floor();
        let mut follower = PathFollower::new(path(), FollowerSettings::default(), 7);
        let directive = follower.tick(
            &world,
            FollowerFrame {
                position: Vec3::new(0.5, 64.0, 0.5),
                on_ground: true,
                horizontal_collision: false,
                paused: false,
            },
        );
        assert!(
            matches!(directive, FollowerDirective::Move { .. }),
            "gave up on a path it was actually on: {directive:?}"
        );
    }
}

#[cfg(test)]
mod yaw_wrap_tests {
    use super::*;

    /// Steering must not accumulate revolutions.
    ///
    /// Turning the same way for long enough used to walk the yaw off to
    /// thousands of degrees, which still points the right way but spends a
    /// float32's precision on whole turns.
    #[test]
    fn steering_stays_within_one_turn_however_far_it_has_turned() {
        let settings = FollowerSettings::default();
        let mut yaw = 0.0f32;
        // Circle the origin, which turns the bot continuously one way.
        for step in 0..4_000 {
            let angle = f64::from(step) * 0.05;
            let target = Vec3::new(angle.cos() * 10.0, 64.0, angle.sin() * 10.0);
            let (next, _) = steering_direction(
                Vec3::new(0.0, 64.0, 0.0),
                yaw,
                0.0,
                target,
                0.0,
                0.0,
                180.0,
                &settings,
            );
            yaw = next;
            assert!(
                (-180.0..=180.0).contains(&yaw),
                "yaw left its range at step {step}: {yaw}"
            );
        }
    }

    /// Wrapping must not change where the bot is actually looking.
    #[test]
    fn wrapping_preserves_the_heading() {
        for raw in [-12187.0f32, -540.0, -181.0, 0.0, 181.0, 359.0, 721.0] {
            let wrapped = wrap_degrees(raw);
            assert!((-180.0..=180.0).contains(&wrapped), "{raw} -> {wrapped}");
            assert!(
                angle_delta(raw, wrapped).abs() < 1e-2,
                "{raw} -> {wrapped} changed the heading"
            );
        }
    }
}
