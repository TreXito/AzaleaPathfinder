//! Tick-driven path execution independent of client state.

use azalea::{BlockPos, Vec3};

use super::moves::{LavaPolicy, lava_risk, lava_transition_allowed};
use super::world::{BlockKind, WorldView};
use crate::{Path, PathNode};

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
            total_ticks: 0,
            sprint_pause: 0,
            escape_hop_a,
            escape_hop_b,
            yaw_phase,
            pitch_phase,
            rng,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn current_node_index(&self) -> usize {
        self.idx
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
        let steer_idx = furthest_visible(world, frame.position, nodes, self.idx, &self.settings);
        let target = node_center(nodes[steer_idx].pos);

        if self.idx > self.last_progress_idx {
            self.last_progress_idx = self.idx;
            self.stalled = 0;
        } else {
            self.stalled = self.stalled.saturating_add(1);
            if self.stalled > self.settings.stall_ticks.max(1) {
                return FollowerDirective::Stuck {
                    at: BlockPos::from(&frame.position),
                };
            }
        }

        let jump = frame.on_ground
            && frame.horizontal_collision
            && (target.y - frame.position.y > self.settings.jump_height_threshold.max(0.0)
                || self.stalled == self.escape_hop_a
                || self.stalled == self.escape_hop_b);
        let sprint = if self.sprint_pause > 0 {
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
        FollowerDirective::Move {
            target,
            sprint,
            jump,
            yaw_bias: (t * self.settings.yaw_wander_speed + self.yaw_phase).sin()
                * self.settings.yaw_wander_degrees,
            pitch_bias: (t * self.settings.pitch_wander_speed + self.pitch_phase).sin()
                * self.settings.pitch_wander_degrees,
            max_turn: self.max_turn,
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
    let desired_yaw = f64::atan2(-dx, dz).to_degrees() as f32 + yaw_bias;
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
        current_yaw + (angle_delta(current_yaw, desired_yaw) * ease).clamp(-max_turn, max_turn),
        current_pitch + ((desired_pitch - current_pitch) * ease).clamp(-max_turn, max_turn),
    )
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
        if node.pos.y != y || !line_walkable(world, position, node_center(node.pos), y, settings) {
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
        let center = BlockPos::new(cx.floor() as i32, feet_y, cz.floor() as i32);
        if !world.standable(center) || position_lava_risk(world, center, settings.lava_policy) > 0 {
            return false;
        }
        for side in [
            settings.body_half_width.max(0.0),
            -settings.body_half_width.max(0.0),
        ] {
            let x = (cx + perp_x * side).floor() as i32;
            let z = (cz + perp_z * side).floor() as i32;
            // Steps block the body's edge at foot height, but not at head height.
            if matches!(
                world.block(BlockPos::new(x, feet_y, z)),
                BlockKind::Solid | BlockKind::Fence | BlockKind::Step
            ) || matches!(
                world.block(BlockPos::new(x, feet_y + 1, z)),
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

        world.0.remove(&(2, 63, 1));
        world.0.insert((2, 64, 1), BlockKind::Step);
        assert!(!line_walkable(
            &world,
            Vec3::new(0.5, 64.0, 0.8),
            Vec3::new(4.5, 64.0, 0.8),
            64,
            &FollowerSettings::default()
        ));
    }
}
