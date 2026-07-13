use azalea::BlockPos;

use super::world::{BlockKind, WorldView, offset};
use crate::types::{Cost, MoveKind};

/// Tuning knobs for a single path search.
pub struct MoveContext {
    /// Maximum blocks the bot may drop in one fall move.
    pub max_fall: i32,
    /// Manhattan distance at which the goal counts as reached.
    pub goal_tolerance: i32,
    /// Node-expansion budget before the search gives up. Deep Caverns
    /// navigation is 3D and the heuristic ignores vertical distance, so the
    /// search fans out — this needs to be generous. Build in release mode
    /// (azalea warns about debug perf) so a large search stays fast.
    pub max_expansions: usize,
    /// Wall-clock budget for one search, in ms. Bounds the worst case — an
    /// unreachable goal otherwise explores the entire loaded cave system
    /// before concluding "no path", which can take seconds in debug builds.
    pub time_budget_ms: u64,
    /// Extra cost per adjacent wall when standing at a node. Real players
    /// leave clearance unless squeezing actually saves distance; 0 turns
    /// the behavior off. (Walking one block costs 10 for scale.)
    pub wall_penalty: Cost,
    /// Extra cost for a node that touches lava (feet/head/floor). Large so
    /// lava is a genuine last resort — the default ~200 blocks-equivalent
    /// means any dry detour up to that length wins. 0 turns it off.
    pub lava_penalty: Cost,

    pub lava_proximity_penalty: Cost,

    pub path_seed: u64,
}

impl Default for MoveContext {
    fn default() -> Self {
        Self {
            max_fall: 3,
            goal_tolerance: 1,
            max_expansions: 150_000,
            time_budget_ms: 2_000,
            wall_penalty: 4,
            lava_penalty: 2000,
            lava_proximity_penalty: 8,
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

/// A movement capability the planner can use. Adding a new way of getting
/// around (parkour jumps, AOTV, etherwarp, ...) means implementing this
/// trait — the A* core never changes.
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
}

/// The standard legit move set: walk, jump up one, fall off ledges, plus
/// the (currently inert) item-teleport extension points.
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

/// Extra cost for standing beside walls (feet or head level), so paths
/// drift toward open space like a player's would. Symmetric situations
/// (1-wide tunnels) penalize every option equally and stay unaffected.
pub fn wall_proximity_penalty(pos: BlockPos, world: &dyn WorldView, per_wall: Cost) -> Cost {
    if per_wall == 0 {
        return 0;
    }
    let is_wall = |b: BlockKind| matches!(b, BlockKind::Solid | BlockKind::Fence);
    let mut penalty = 0;
    for (dx, dz) in CARDINALS {
        if is_wall(world.block(offset(pos, dx, 0, dz)))
            || is_wall(world.block(offset(pos, dx, 1, dz)))
        {
            penalty += per_wall;
        }
    }
    penalty
}

/// Cost added for standing at a node that touches lava — feet or head in
/// it, or standing on its surface. Flat (not per-contact) so it's a clean
/// "was any lava involved in this step" toll that makes lava a last resort.
pub fn lava_penalty(pos: BlockPos, world: &dyn WorldView, per: Cost) -> Cost {
    if per == 0 {
        return 0;
    }
    let touches_lava = world.block(pos) == BlockKind::Lava
        || world.block(offset(pos, 0, 1, 0)) == BlockKind::Lava
        || world.block(offset(pos, 0, -1, 0)) == BlockKind::Lava;
    if touches_lava { per } else { 0 }
}

pub fn lava_proximity_penalty(pos: BlockPos, world: &dyn WorldView, per_block: Cost) -> Cost {
    if per_block == 0 {
        return 0;
    }

    let mut penalty = 0;

    let radius = 3;

    for dx in -radius..=radius {
        for dz in -radius..=radius {
            if dx == 0 && dz == 0 {
                continue;
            }

            for dy in -radius..=radius {
                if world.block(offset(pos, dx, dy, dz)) == BlockKind::Lava {
                    penalty += per_block;
                    break;
                }
            }
        }
    }

    penalty
}

/// Step one block on level ground — cardinal, or diagonal when neither
/// flanking cell clips the player's body (humans cut corners, not blocks).
pub struct WalkMove;

impl Move for WalkMove {
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        _ctx: &MoveContext,
        out: &mut Vec<Edge>,
    ) {
        for (dx, dz) in CARDINALS {
            let to = offset(from, dx, 0, dz);
            if world.standable(to) {
                out.push(Edge {
                    to,
                    kind: MoveKind::Walk,
                    cost: 10,
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
                // ~10 * sqrt(2)
                out.push(Edge {
                    to,
                    kind: MoveKind::Walk,
                    cost: 14,
                });
            }
        }
        // stairs and slabs: half-block rises/drops are WALKED (physics
        // auto-step), never jumped. Ascending is only a walk when we're
        // already standing inside a step cell (elevated half a block);
        // reaching a step from flat ground is a same-y move, and a step
        // sitting a full block up still needs the jump move.
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
                    cost: 12,
                });
            }
        }
    }
}

/// Jump up a single block step in a cardinal direction.
pub struct JumpMove;

impl Move for JumpMove {
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        _ctx: &MoveContext,
        out: &mut Vec<Edge>,
    ) {
        // headroom above the current position to jump at all
        if world.block(offset(from, 0, 2, 0)) != BlockKind::Air {
            return;
        }
        for (dx, dz) in CARDINALS {
            let to = offset(from, dx, 1, dz);

            if world.standable(to) {
                out.push(Edge {
                    to,
                    kind: MoveKind::Jump,
                    cost: 14,
                });
            }
        }
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
            // need clearance to walk into the gap, and it must be a gap
            if world.block(step) != BlockKind::Air
                || world.block(offset(step, 0, 1, 0)) != BlockKind::Air
                || world.standable(step)
            {
                continue;
            }
            // scan straight down for a landing spot (solid floor or a
            // slab/stair cell both count as landing)
            for drop in 1..=ctx.max_fall {
                let feet = offset(step, 0, -drop, 0);
                if world.standable(feet) {
                    out.push(Edge {
                        to: feet,
                        kind: MoveKind::Fall,
                        cost: 10 + 3 * drop as Cost,
                    });
                    break;
                }
                if world.block(feet) != BlockKind::Air {
                    break; // hit something unlandable before finding floor
                }
            }
        }
    }
}
