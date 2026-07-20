use azalea::BlockPos;

use super::world::{BlockKind, WorldView, offset};
use crate::types::{Cost, MoveKind};

/// Controls whether the planner may enter lava or its safety buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LavaPolicy {
    /// Never enter the clearance buffer. If the start is already unsafe, only
    /// transitions that strictly reduce lava exposure are permitted.
    Forbidden { clearance: i32 },
    /// Allow lava, with the configured penalties added to the route.
    Penalized,
}

impl Default for LavaPolicy {
    fn default() -> Self {
        Self::Forbidden { clearance: 2 }
    }
}

/// Movement costs in tenths of a flat block.
#[derive(Debug, Clone, Copy)]
pub struct MovementCosts {
    pub cardinal_walk: Cost,
    pub diagonal_walk: Cost,
    pub step: Cost,
    pub jump: Cost,
    pub fall_base: Cost,
    pub fall_per_block: Cost,
}

impl Default for MovementCosts {
    fn default() -> Self {
        Self {
            cardinal_walk: 10,
            diagonal_walk: 14,
            step: 12,
            jump: 24,
            fall_base: 14,
            fall_per_block: 6,
        }
    }
}

/// Tuning knobs for a single path search.
#[derive(Debug, Clone)]
pub struct MoveContext {
    /// Maximum blocks the bot may drop in one fall move.
    pub max_fall: i32,
    /// Manhattan distance at which the goal counts as reached.
    pub goal_tolerance: i32,
    /// Maximum nodes expanded during one search.
    pub max_expansions: usize,
    /// Wall-clock limit for one search, in milliseconds.
    pub time_budget_ms: u64,
    /// Base cost for each movement type.
    pub costs: MovementCosts,
    /// Extra cost per adjacent wall. Set to zero to disable wall clearance.
    pub wall_penalty: Cost,
    /// Whether lava and its clearance buffer are forbidden or merely costly.
    pub lava_policy: LavaPolicy,
    /// Direct-contact toll used only by [`LavaPolicy::Penalized`].
    pub lava_penalty: Cost,
    /// Radius for the soft avoidance cost beyond the hard clearance buffer.
    pub lava_proximity_radius: i32,
    /// Cost multiplier for nearby lava outside the hard buffer.
    pub lava_proximity_penalty: Cost,

    /// Seed used to vary ties between equal-cost paths.
    pub path_seed: u64,
}

impl Default for MoveContext {
    fn default() -> Self {
        Self {
            max_fall: 3,
            goal_tolerance: 1,
            max_expansions: 150_000,
            time_budget_ms: 2_000,
            costs: MovementCosts::default(),
            wall_penalty: 2,
            lava_policy: LavaPolicy::default(),
            lava_penalty: 10_000,
            lava_proximity_radius: 4,
            lava_proximity_penalty: 6,
            path_seed: 0,
        }
    }
}

/// A candidate transition produced by a movement rule.
pub struct Edge {
    pub to: BlockPos,
    pub kind: MoveKind,
    pub cost: Cost,
}

/// A movement rule used by the planner.
pub trait Move: Send + Sync {
    /// Push every position reachable from `from` in one application of
    /// this move onto `out`.
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        ctx: &MoveContext,
        out: &mut Vec<Edge>,
    );

    /// Whether this rule satisfies the built-in heuristic's distance and cost
    /// assumptions. Return `false` for long-range or unusually cheap moves.
    fn supports_builtin_heuristic(&self) -> bool {
        false
    }
}

/// Standard walking, jumping, falling, and teleport extension points.
pub fn default_moves() -> Vec<Box<dyn Move>> {
    vec![
        Box::new(WalkMove),
        Box::new(JumpMove),
        Box::new(FallMove),
        Box::new(super::teleport::AotvMove),
        Box::new(super::teleport::EtherwarpMove),
    ]
}

const CARDINALS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAGONALS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Adds a cost for each adjacent wall at feet or head height.
pub fn wall_proximity_penalty(pos: BlockPos, world: &dyn WorldView, per_wall: Cost) -> Cost {
    if per_wall == 0 {
        return 0;
    }
    let is_wall = |b: BlockKind| matches!(b, BlockKind::Solid | BlockKind::Fence);
    let mut penalty: Cost = 0;
    for (dx, dz) in CARDINALS {
        if is_wall(world.block(offset(pos, dx, 0, dz)))
            || is_wall(world.block(offset(pos, dx, 1, dz)))
        {
            penalty += per_wall;
        }
    }
    penalty
}

/// Adds a flat cost when the player's feet, head, or floor touches lava.
pub fn lava_penalty(pos: BlockPos, world: &dyn WorldView, per: Cost) -> Cost {
    if per == 0 {
        return 0;
    }
    let touches_lava = world.block(pos) == BlockKind::Lava
        || world.block(offset(pos, 0, 1, 0)) == BlockKind::Lava
        || world.block(offset(pos, 0, -1, 0)) == BlockKind::Lava;
    if touches_lava { per } else { 0 }
}

/// Returns lava exposure within `clearance`; zero means safe.
pub fn lava_risk(pos: BlockPos, world: &dyn WorldView, clearance: i32) -> Cost {
    let clearance = clearance.max(0);
    let mut nearest: Option<i32> = None;
    for dx in -clearance..=clearance {
        for dz in -clearance..=clearance {
            let distance = dx.abs().max(dz.abs());
            if nearest.is_some_and(|best| distance >= best) {
                continue;
            }
            for dy in -1..=1 {
                if world.block(offset(pos, dx, dy, dz)) == BlockKind::Lava {
                    nearest = Some(distance);
                    break;
                }
            }
        }
    }
    nearest
        .map(|distance| (clearance + 1 - distance) as Cost)
        .unwrap_or(0)
}

/// Safe nodes may only enter safe nodes. A start already inside the buffer can
/// escape, but every step must strictly lower its exposure.
pub fn lava_transition_allowed(from_risk: Cost, to_risk: Cost, policy: LavaPolicy) -> bool {
    match policy {
        LavaPolicy::Penalized => true,
        LavaPolicy::Forbidden { .. } => to_risk == 0 || (from_risk > 0 && to_risk < from_risk),
    }
}

pub fn lava_proximity_penalty(
    pos: BlockPos,
    world: &dyn WorldView,
    radius: i32,
    per_block: Cost,
) -> Cost {
    if per_block == 0 || radius <= 0 {
        return 0;
    }

    // Use the nearest lava so large pools do not multiply the cost.
    for distance in 1..=radius {
        for dx in -distance..=distance {
            for dz in -distance..=distance {
                if dx.abs().max(dz.abs()) != distance {
                    continue;
                }
                for dy in -1..=1 {
                    if world.block(offset(pos, dx, dy, dz)) == BlockKind::Lava {
                        let weight = (radius + 1 - distance) as Cost;
                        return per_block.saturating_mul(weight);
                    }
                }
            }
        }
    }

    0
}

/// Walks on level ground, stairs, and slabs. Diagonals require clear corners.
pub struct WalkMove;

impl Move for WalkMove {
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        ctx: &MoveContext,
        out: &mut Vec<Edge>,
    ) {
        for (dx, dz) in CARDINALS {
            let to = offset(from, dx, 0, dz);
            if world.standable(to) {
                out.push(Edge {
                    to,
                    kind: MoveKind::Walk,
                    cost: ctx.costs.cardinal_walk,
                });
            }
        }
        let body_clear = |p: BlockPos| {
            world.block(p) == BlockKind::Air && world.block(offset(p, 0, 1, 0)) == BlockKind::Air
        };
        for (dx, dz) in DIAGONALS {
            let to = offset(from, dx, 0, dz);
            if world.standable(to)
                && body_clear(offset(from, dx, 0, 0))
                && body_clear(offset(from, 0, 0, dz))
            {
                out.push(Edge {
                    to,
                    kind: MoveKind::Walk,
                    cost: ctx.costs.diagonal_walk,
                });
            }
        }
        // Auto-step handles stairs and slabs. A full-block rise still uses
        // JumpMove unless the player is already standing on a raised step.
        let from_step = world.block(from) == BlockKind::Step;
        for (dx, dz) in CARDINALS {
            for dy in [1, -1] {
                let to = offset(from, dx, dy, dz);
                if !world.standable(to) {
                    continue;
                }
                let to_step = world.block(to) == BlockKind::Step;
                let walkable = match dy {
                    1 => from_step,
                    _ => from_step || to_step,
                };
                if !walkable {
                    continue;
                }
                if dy == 1 && world.block(offset(from, 0, 2, 0)) != BlockKind::Air {
                    continue; // no headroom to rise
                }
                out.push(Edge {
                    to,
                    kind: MoveKind::Walk,
                    cost: ctx.costs.step,
                });
            }
        }
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }
}

/// Jump up a single block step in a cardinal direction.
pub struct JumpMove;

impl Move for JumpMove {
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        ctx: &MoveContext,
        out: &mut Vec<Edge>,
    ) {
        if world.block(offset(from, 0, 2, 0)) != BlockKind::Air {
            return;
        }
        for (dx, dz) in CARDINALS {
            let to = offset(from, dx, 1, dz);

            if world.standable(to) {
                out.push(Edge {
                    to,
                    kind: MoveKind::Jump,
                    cost: ctx.costs.jump,
                });
            }
        }
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }
}

/// Walk off an edge and drop up to `ctx.max_fall` blocks.
pub struct FallMove;

impl Move for FallMove {
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        ctx: &MoveContext,
        out: &mut Vec<Edge>,
    ) {
        for (dx, dz) in CARDINALS {
            let step = offset(from, dx, 0, dz);
            if world.block(step) != BlockKind::Air
                || world.block(offset(step, 0, 1, 0)) != BlockKind::Air
                || world.standable(step)
            {
                continue;
            }
            // Stop at the first valid landing or obstruction.
            for drop in 1..=ctx.max_fall {
                let feet = offset(step, 0, -drop, 0);
                if world.standable(feet) {
                    out.push(Edge {
                        to: feet,
                        kind: MoveKind::Fall,
                        cost: ctx
                            .costs
                            .fall_base
                            .saturating_add(ctx.costs.fall_per_block.saturating_mul(drop as Cost)),
                    });
                    break;
                }
                if world.block(feet) != BlockKind::Air {
                    break; // hit something unlandable before finding floor
                }
            }
        }
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_policy_only_allows_risk_reducing_escape_steps() {
        let policy = LavaPolicy::Forbidden { clearance: 2 };

        assert!(lava_transition_allowed(0, 0, policy));
        assert!(!lava_transition_allowed(0, 1, policy));
        assert!(lava_transition_allowed(3, 2, policy));
        assert!(!lava_transition_allowed(3, 3, policy));
        assert!(!lava_transition_allowed(2, 3, policy));
    }
}
