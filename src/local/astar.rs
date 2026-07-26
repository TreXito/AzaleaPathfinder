use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use azalea::BlockPos;

use super::moves::{Edge, Move, MoveContext};
use super::world::WorldView;
use crate::types::{Cost, MoveKind, Path, PathError, PathNode};

type Key = (i32, i32, i32);

fn cached_lava_risk(
    pos: BlockPos,
    world: &dyn WorldView,
    ctx: &MoveContext,
    risks: &mut HashMap<Key, Cost>,
) -> Cost {
    let super::moves::LavaPolicy::Forbidden { clearance } = ctx.lava_policy else {
        return 0;
    };
    *risks
        .entry((pos.x, pos.y, pos.z))
        .or_insert_with(|| super::moves::lava_risk(pos, world, clearance))
}

fn lava_edge_allowed(
    from: BlockPos,
    to: BlockPos,
    world: &dyn WorldView,
    ctx: &MoveContext,
    risks: &mut HashMap<Key, Cost>,
) -> bool {
    let super::moves::LavaPolicy::Forbidden { .. } = ctx.lava_policy else {
        return true;
    };
    let from_risk = cached_lava_risk(from, world, ctx, risks);
    let to_risk = cached_lava_risk(to, world, ctx, risks);
    super::moves::lava_transition_allowed(from_risk, to_risk, ctx.lava_policy)
}

fn lava_safe_to_finish(risk: Cost, ctx: &MoveContext) -> bool {
    matches!(ctx.lava_policy, super::moves::LavaPolicy::Penalized) || risk == 0
}

fn water_risk(pos: BlockPos, world: &dyn WorldView, ctx: &MoveContext) -> u8 {
    u8::from(
        matches!(ctx.water_policy, super::moves::WaterPolicy::Forbidden)
            && world.block(pos) == super::world::BlockKind::Water,
    )
}

fn terrain_penalty(pos: BlockPos, world: &dyn WorldView, ctx: &MoveContext) -> Cost {
    let direct_lava = if matches!(ctx.lava_policy, super::moves::LavaPolicy::Penalized) {
        super::moves::lava_penalty(pos, world, ctx.lava_penalty)
    } else {
        0
    };
    super::moves::wall_proximity_penalty(pos, world, ctx.wall_penalty)
        .saturating_add(direct_lava)
        .saturating_add(super::moves::lava_proximity_penalty(
            pos,
            world,
            ctx.lava_proximity_radius,
            ctx.lava_proximity_penalty,
        ))
        .saturating_add(ctx.avoid.get(&pos).copied().unwrap_or(0))
        .saturating_add(stepping_stone_penalty(pos, world, ctx))
        .saturating_add(grazing_step_penalty(pos, world, ctx))
}

/// Toll for standing on a partial block that has a taller partial block beside
/// it, where a corner of the body can be lifted without the bot meaning to.
///
/// A body is 0.6 wide in a 1.0 block, so walking anywhere but the exact middle
/// leaves part of the hitbox over a neighbouring column - and a player stands
/// on the tallest thing any part of it overlaps. Over a field of snow whose
/// depth changes block to block, that means being lifted a fraction of a block
/// by a corner, repeatedly, without ever deliberately stepping up.
///
/// The client and the server do not always resolve that same corner the same
/// way. Every anticheat Simulation flag in a measured summit climb landed on
/// exactly this: a step-up onto a fractional height, off by a quarter of a
/// block every time, with both sides holding identical block data. One of the
/// flagged spots had the body overlapping the taller column by five hundredths
/// of a block.
///
/// A toll rather than a ban, and not a large one: a snowy mountain is made of
/// these, so refusing them outright would make the summit unreachable. Priced
/// at a few blocks of walking, it keeps the route down the middle of an even
/// field and across the flatter side of an uneven one, and still crosses when
/// there is no way round.
fn grazing_step_penalty(pos: BlockPos, world: &dyn WorldView, ctx: &MoveContext) -> Cost {
    use super::world::BlockKind;
    let BlockKind::Step(here) = world.block(pos) else {
        return 0;
    };
    // All eight neighbours, because the corners are the whole problem: a
    // cardinal neighbour is something the bot walks squarely onto and both
    // sides agree about.
    for dx in -1..=1 {
        for dz in -1..=1 {
            if (dx, dz) == (0, 0) {
                continue;
            }
            if let BlockKind::Step(there) = world.block(super::world::offset(pos, dx, 0, dz))
                && there > here
            {
                return ctx.grazing_step_penalty;
            }
        }
    }
    0
}

/// Toll for standing on something held up by water: a lily pad.
///
/// A pad is genuinely standable and must stay so, or a bot that lands on one is
/// marooned with no legal move. But a route that hops pads across a lake is one
/// overshoot away from the water, and with water forbidden the water is a dead
/// end the bot cannot plan its way out of: the hub lake failed a summit run
/// twice this way. Priced like a swim, so pads are used only when there is no
/// way round, which is what the water penalty already means.
fn stepping_stone_penalty(pos: BlockPos, world: &dyn WorldView, ctx: &MoveContext) -> Cost {
    if !matches!(ctx.water_policy, super::moves::WaterPolicy::Forbidden) {
        return 0;
    }
    let below = super::world::offset(pos, 0, -1, 0);
    if matches!(world.block(pos), super::world::BlockKind::Step(_))
        && world.block(below) == super::world::BlockKind::Water
    {
        ctx.water_penalty
    } else {
        0
    }
}

struct NodeData {
    g: Cost,
    parent: Option<BlockPos>,
    reached_by: MoveKind,
}

/// Lower bound for A*. Horizontal and vertical estimates are combined with
/// `max` because one move can advance on both axes.
fn heuristic(pos: BlockPos, goal: BlockPos, ctx: &MoveContext) -> Cost {
    // Applying tolerance to each axis is looser than Manhattan tolerance, so
    // the estimate stays admissible.
    let tolerance = ctx.goal_tolerance.max(0) as u32;
    let dx = pos.x.abs_diff(goal.x).saturating_sub(tolerance);
    let dz = pos.z.abs_diff(goal.z).saturating_sub(tolerance);
    let dy = pos.y.abs_diff(goal.y).saturating_sub(tolerance);
    // Every built-in move that can make horizontal progress participates here.
    // Using Chebyshev distance keeps diagonal parkour admissible even when a
    // caller configures unusually cheap movement costs.
    let horizontal_per_axis = ctx
        .costs
        .cardinal_walk
        .min(ctx.costs.diagonal_walk)
        .min(ctx.costs.step)
        .min(ctx.costs.jump)
        .min(ctx.costs.fall_base.saturating_add(ctx.costs.fall_per_block))
        .min(ctx.costs.parkour_per_block)
        .min(ctx.costs.swim)
        .min(ctx.costs.swim_exit)
        .min(ctx.costs.climb);
    let horiz = horizontal_per_axis.saturating_mul(dx.max(dz));
    let vert = if pos.y < goal.y {
        ctx.costs
            .step
            .min(ctx.costs.jump)
            .min(ctx.costs.climb)
            .min(ctx.costs.swim)
            .min(ctx.costs.swim_exit)
            .saturating_mul(dy)
    } else {
        ctx.costs
            .step
            .min(ctx.costs.climb)
            .min(ctx.costs.swim)
            // Descending parkour pays this per dropped block but no fall base.
            .min(ctx.costs.fall_per_block)
            .saturating_mul(dy)
    };
    horiz.max(vert)
}

/// Finds a path to within `ctx.goal_tolerance` Manhattan distance of `goal`.
/// Returns an error instead of a partial path when the goal cannot be reached.
pub fn find_path(
    world: &dyn WorldView,
    start: BlockPos,
    goal: BlockPos,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
) -> Result<Path, PathError> {
    search(world, start, goal, moves, ctx, SearchMode::Strict).map(|(path, _)| path)
}

/// Finds a complete path when possible, or the best reachable partial path.
/// The boolean is `true` only when the goal was reached. The path always starts
/// at `start`; if no progress is possible, it contains only that node.
pub fn find_path_best_effort(
    world: &dyn WorldView,
    start: BlockPos,
    goal: BlockPos,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
) -> (Path, bool) {
    match search(world, start, goal, moves, ctx, SearchMode::BestEffort) {
        Ok(result) => result,
        Err(_) => unreachable!("best-effort search always returns a partial path"),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    Strict,
    BestEffort,
}

/// Shared A* implementation for strict and partial-path callers.
fn search(
    world: &dyn WorldView,
    start: BlockPos,
    goal: BlockPos,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
    mode: SearchMode,
) -> Result<(Path, bool), PathError> {
    let use_heuristic = moves
        .iter()
        .all(|movement| movement.supports_builtin_heuristic());
    let h = |p: BlockPos| -> Cost {
        if use_heuristic {
            heuristic(p, goal, ctx)
        } else {
            0
        }
    };
    let manhattan = |p: BlockPos| -> u64 {
        u64::from(p.x.abs_diff(goal.x))
            + u64::from(p.y.abs_diff(goal.y))
            + u64::from(p.z.abs_diff(goal.z))
    };
    let goal_tolerance = u64::from(ctx.goal_tolerance.max(0) as u32);
    let key = |p: BlockPos| -> Key { (p.x, p.y, p.z) };

    let mut nodes: HashMap<Key, NodeData> = HashMap::new();
    let mut open: BinaryHeap<Reverse<(Cost, Cost, Key)>> = BinaryHeap::new();
    nodes.insert(
        key(start),
        NodeData {
            g: 0,
            parent: None,
            reached_by: MoveKind::Start,
        },
    );
    open.push(Reverse((
        h(start),
        tie_break(start, ctx.path_seed),
        key(start),
    )));

    let mut lava_risks: HashMap<Key, Cost> = HashMap::new();
    // When starting in a forbidden hazard, escaping it takes priority over
    // goal distance. Lava remains first because it is immediately damaging.
    let mut best_lava_risk = cached_lava_risk(start, world, ctx, &mut lava_risks);
    let mut best_water_risk = water_risk(start, world, ctx);
    let mut best_dist = manhattan(start);
    let mut best = (start, 0u32);

    let mut expansions = 0usize;
    let mut scratch: Vec<Edge> = Vec::new();
    let mut terrain_costs: HashMap<Key, Cost> = HashMap::new();
    let deadline =
        std::time::Instant::now().checked_add(std::time::Duration::from_millis(ctx.time_budget_ms));

    while let Some(Reverse((f, _, k))) = open.pop() {
        let pos = BlockPos::new(k.0, k.1, k.2);
        let g = nodes[&k].g;
        if f > g.saturating_add(h(pos)) {
            continue;
        }
        let current_lava_risk = cached_lava_risk(pos, world, ctx, &mut lava_risks);
        let current_water_risk = water_risk(pos, world, ctx);
        if manhattan(pos) <= goal_tolerance
            && lava_safe_to_finish(current_lava_risk, ctx)
            && current_water_risk == 0
        {
            return Ok((reconstruct(&nodes, pos, g), true));
        }
        let md = manhattan(pos);
        if current_lava_risk < best_lava_risk
            || (current_lava_risk == best_lava_risk
                && (current_water_risk < best_water_risk
                    || (current_water_risk == best_water_risk && md < best_dist)))
        {
            best_lava_risk = current_lava_risk;
            best_water_risk = current_water_risk;
            best_dist = md;
            best = (pos, g);
        }
        expansions += 1;
        if expansions > ctx.max_expansions {
            if mode == SearchMode::Strict {
                return Err(PathError::SearchBudgetExhausted);
            }
            break;
        }
        if (expansions == 1 || expansions.is_multiple_of(512))
            && deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
        {
            if mode == SearchMode::Strict {
                return Err(PathError::SearchBudgetExhausted);
            }
            break;
        }
        scratch.clear();
        for m in moves {
            m.candidates(pos, world, ctx, &mut scratch);
        }
        for edge in scratch.drain(..) {
            if !lava_edge_allowed(pos, edge.to, world, ctx, &mut lava_risks) {
                continue;
            }
            let ek = key(edge.to);
            let base_g = g.saturating_add(edge.cost);
            if nodes.get(&ek).is_some_and(|nd| base_g >= nd.g) {
                continue;
            }
            let terrain = *terrain_costs
                .entry(ek)
                .or_insert_with(|| terrain_penalty(edge.to, world, ctx));
            let ng = base_g.saturating_add(terrain);
            let better = nodes.get(&ek).is_none_or(|nd| ng < nd.g);
            if better {
                nodes.insert(
                    ek,
                    NodeData {
                        g: ng,
                        parent: Some(pos),
                        reached_by: edge.kind,
                    },
                );
                open.push(Reverse((
                    ng.saturating_add(h(edge.to)),
                    tie_break(edge.to, ctx.path_seed),
                    ek,
                )));
            }
        }
    }

    if mode == SearchMode::Strict {
        return Err(PathError::NoPath);
    }
    // A partial path lets the caller move, load more chunks, and plan again.
    Ok((reconstruct(&nodes, best.0, best.1), false))
}

fn reconstruct(nodes: &HashMap<Key, NodeData>, end: BlockPos, total_cost: Cost) -> Path {
    let mut out = Vec::new();
    let mut at = Some(end);
    while let Some(pos) = at {
        let nd = &nodes[&(pos.x, pos.y, pos.z)];
        out.push(PathNode {
            pos,
            reached_by: nd.reached_by,
        });
        at = nd.parent;
    }
    out.reverse();
    Path {
        nodes: out,
        total_cost,
    }
}

fn tie_break(pos: BlockPos, seed: u64) -> Cost {
    let mut x = seed
        ^ (pos.x as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (pos.z as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ (pos.y as u64);

    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;

    (x % 3) as Cost
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::super::moves::default_moves;
    use super::super::world::BlockKind;
    use super::*;

    /// Small test world where unspecified blocks are air.
    struct Grid {
        solid: HashSet<(i32, i32, i32)>,
        steps: std::collections::HashMap<(i32, i32, i32), u8>,
        lava: HashSet<(i32, i32, i32)>,
        water: HashSet<(i32, i32, i32)>,
    }

    impl Grid {
        fn new() -> Self {
            Self {
                solid: HashSet::new(),
                steps: std::collections::HashMap::new(),
                lava: HashSet::new(),
                water: HashSet::new(),
            }
        }

        /// Fills `ys` with water, clearing anything solid already there.
        fn pool(
            &mut self,
            xs: std::ops::RangeInclusive<i32>,
            zs: std::ops::RangeInclusive<i32>,
            ys: std::ops::RangeInclusive<i32>,
        ) {
            for x in xs {
                for z in zs.clone() {
                    for y in ys.clone() {
                        self.solid.remove(&(x, y, z));
                        self.water.insert((x, y, z));
                    }
                }
            }
        }

        /// Adds a partial block of the given height, in sixteenths.
        fn step(&mut self, x: i32, y: i32, z: i32, height: u8) {
            self.steps.insert((x, y, z), height);
        }

        /// Adds a solid floor; the player stands one block above it.
        fn floor(
            &mut self,
            xs: std::ops::RangeInclusive<i32>,
            zs: std::ops::RangeInclusive<i32>,
            y: i32,
        ) {
            for x in xs {
                for z in zs.clone() {
                    self.solid.insert((x, y, z));
                }
            }
        }
    }

    impl WorldView for Grid {
        fn block(&self, pos: BlockPos) -> BlockKind {
            let k = (pos.x, pos.y, pos.z);
            if self.solid.contains(&k) {
                BlockKind::Solid
            } else if let Some(height) = self.steps.get(&k) {
                BlockKind::Step(*height)
            } else if self.lava.contains(&k) {
                BlockKind::Lava
            } else if self.water.contains(&k) {
                BlockKind::Water
            } else {
                BlockKind::Air
            }
        }
    }

    fn ctx() -> MoveContext {
        MoveContext::default()
    }

    /// Context for the tests that are about *how* the bot swims rather than
    /// whether it should. The default policy forbids water outright.
    fn wet_ctx() -> MoveContext {
        MoveContext {
            water_policy: super::super::moves::WaterPolicy::Penalized,
            water_penalty: 0,
            ..MoveContext::default()
        }
    }

    fn dist(a: BlockPos, b: BlockPos) -> i32 {
        (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs()
    }

    struct CheapDetourMove;

    impl Move for CheapDetourMove {
        fn candidates(
            &self,
            from: BlockPos,
            _world: &dyn WorldView,
            _ctx: &MoveContext,
            out: &mut Vec<Edge>,
        ) {
            let start = BlockPos::new(0, 64, 0);
            let detour = BlockPos::new(-100, 64, 0);
            let goal = BlockPos::new(10, 64, 0);
            if from == start {
                out.push(Edge {
                    to: detour,
                    kind: MoveKind::Jump,
                    cost: 1,
                });
            } else if from == detour {
                out.push(Edge {
                    to: goal,
                    kind: MoveKind::Jump,
                    cost: 1,
                });
            }
        }
    }

    #[test]
    fn custom_long_range_moves_fall_back_to_optimal_dijkstra_search() {
        let mut grid = Grid::new();
        grid.floor(-2..=12, -1..=1, 63);
        let mut moves = default_moves();
        moves.push(Box::new(CheapDetourMove));
        let ctx = MoveContext {
            goal_tolerance: 0,
            ..ctx()
        };

        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(10, 64, 0),
            &moves,
            &ctx,
        )
        .unwrap();
        assert_eq!(path.total_cost, 2);
        assert_eq!(path.nodes[1].pos, BlockPos::new(-100, 64, 0));
    }

    #[test]
    fn walks_across_flat_floor() {
        let mut grid = Grid::new();
        grid.floor(-2..=8, -2..=2, 63);

        let goal = BlockPos::new(5, 64, 0);
        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            goal,
            &default_moves(),
            &ctx(),
        )
        .unwrap();

        assert!(dist(path.nodes.last().unwrap().pos, goal) <= 1);
        assert!(
            path.nodes
                .iter()
                .skip(1)
                .all(|n| n.reached_by == MoveKind::Walk)
        );
    }

    #[test]
    fn jumps_up_a_step() {
        let mut grid = Grid::new();
        grid.floor(-2..=2, -2..=2, 63); // stand at y=64
        grid.floor(3..=8, -2..=2, 64); // stand at y=65

        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(6, 65, 0),
            &default_moves(),
            &ctx(),
        )
        .unwrap();

        assert!(path.nodes.iter().any(|n| n.reached_by == MoveKind::Jump));
    }

    #[test]
    fn falls_off_a_ledge() {
        let mut grid = Grid::new();
        grid.floor(-2..=2, -2..=2, 65); // stand at y=66
        grid.floor(3..=8, -2..=2, 63); // stand at y=64, a 2-block drop

        let path = find_path(
            &grid,
            BlockPos::new(0, 66, 0),
            BlockPos::new(6, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .unwrap();

        assert!(path.nodes.iter().any(|n| n.reached_by == MoveKind::Fall));
    }

    #[test]
    fn walks_up_stairs_without_jumping() {
        let mut grid = Grid::new();
        grid.floor(-2..=2, -1..=1, 63); // lower floor, stand at y=64
        grid.step(3, 64, 0, super::super::world::HALF_BLOCK); // stair
        grid.step(4, 65, 0, super::super::world::HALF_BLOCK); // stair
        grid.floor(5..=8, -1..=1, 65); // upper floor, stand at y=66

        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(7, 66, 0),
            &default_moves(),
            &ctx(),
        )
        .unwrap();

        assert!(
            path.nodes.iter().all(|n| n.reached_by != MoveKind::Jump),
            "stairs were jumped: {:?}",
            path.nodes
                .iter()
                .map(|n| (n.pos, n.reached_by))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn avoids_lava_when_a_dry_route_exists() {
        let mut grid = Grid::new();
        grid.floor(-1..=7, -1..=5, 63); // stand at y=64
        // A safe detour remains outside the two-block lava buffer.
        for z in -1..=1 {
            grid.solid.remove(&(3, 63, z));
            grid.lava.insert((3, 63, z));
        }

        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(6, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .unwrap();

        assert!(
            path.nodes
                .iter()
                .all(|n| super::super::moves::lava_risk(n.pos, &grid, 2) == 0),
            "path entered the lava clearance buffer despite a safe detour: {:?}",
            path.nodes.iter().map(|n| n.pos).collect::<Vec<_>>()
        );
    }

    #[test]
    fn routes_round_a_block_that_a_previous_leg_died_on() {
        let mut grid = Grid::new();
        grid.floor(-1..=7, -1..=5, 63); // stand at y=64

        let straight = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(6, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .unwrap();
        assert!(
            straight
                .nodes
                .iter()
                .any(|n| n.pos == BlockPos::new(3, 64, 0))
        );

        let mut avoid = std::collections::HashMap::new();
        avoid.insert(BlockPos::new(3, 64, 0), 800);
        let round = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(6, 64, 0),
            &default_moves(),
            &MoveContext {
                avoid: std::sync::Arc::new(avoid),
                ..ctx()
            },
        )
        .unwrap();

        assert!(
            round.nodes.iter().all(|n| n.pos != BlockPos::new(3, 64, 0)),
            "took the tolled block anyway: {:?}",
            round.nodes.iter().map(|n| n.pos).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_tolled_block_is_still_used_when_it_is_the_only_way() {
        let mut grid = Grid::new();
        grid.floor(-1..=7, 0..=0, 63); // one-block-wide corridor

        let mut avoid = std::collections::HashMap::new();
        avoid.insert(BlockPos::new(3, 64, 0), 800);
        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(6, 64, 0),
            &default_moves(),
            &MoveContext {
                avoid: std::sync::Arc::new(avoid),
                ..ctx()
            },
        )
        .unwrap();

        assert!(path.nodes.iter().any(|n| n.pos == BlockPos::new(3, 64, 0)));
    }

    #[test]
    fn refuses_lava_even_when_it_is_the_only_route() {
        let mut grid = Grid::new();
        // The only route through this one-block corridor crosses lava.
        grid.floor(-1..=7, 0..=0, 63);
        grid.solid.remove(&(3, 63, 0));
        grid.lava.insert((3, 63, 0));

        let err = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(6, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .unwrap_err();

        assert!(matches!(err, PathError::NoPath));
    }

    #[test]
    fn best_effort_prioritizes_escaping_an_existing_lava_hazard() {
        let mut grid = Grid::new();
        // Escaping east initially moves away from the unreachable western goal.
        grid.floor(0..=4, 0..=0, 63);
        grid.solid.remove(&(0, 63, 0));
        grid.lava.insert((0, 63, 0));
        let start = BlockPos::new(0, 64, 0);
        let goal = BlockPos::new(-6, 64, 0);

        let (path, reached) = find_path_best_effort(&grid, start, goal, &default_moves(), &ctx());

        assert!(!reached, "the westward goal is outside the known floor");
        let risks = path
            .nodes
            .iter()
            .map(|node| super::super::moves::lava_risk(node.pos, &grid, 2))
            .collect::<Vec<_>>();
        assert_eq!(risks.first(), Some(&3));
        assert!(
            risks
                .windows(2)
                .all(|pair| pair[1] == 0 || pair[1] < pair[0]),
            "lava exposure did not strictly decrease until safe: {risks:?}"
        );
        assert_eq!(risks.last(), Some(&0));
        assert!(
            path.nodes.last().unwrap().pos.x >= 3,
            "the partial route should move east until clear: {:?}",
            path.nodes.iter().map(|node| node.pos).collect::<Vec<_>>()
        );
    }

    #[test]
    fn no_path_out_of_a_sealed_box() {
        let mut grid = Grid::new();
        grid.floor(-2..=2, -2..=2, 63);
        // Seal the room with walls and a roof.
        for y in 64..=67 {
            for x in -2..=2 {
                grid.solid.insert((x, y, -2));
                grid.solid.insert((x, y, 2));
            }
            for z in -2..=2 {
                grid.solid.insert((-2, y, z));
                grid.solid.insert((2, y, z));
            }
        }
        grid.floor(-2..=2, -2..=2, 68);

        let err = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(20, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .unwrap_err();

        assert!(matches!(err, PathError::NoPath));
    }

    #[test]
    fn swims_across_a_three_block_water_channel() {
        let mut grid = Grid::new();
        // A flooded stretch of a two-high corridor. The low ceiling is what
        // rules out jumping it: vanilla will not jump with a block over the
        // head, so the water has to be swum.
        grid.floor(-2..=8, -1..=1, 63);
        grid.floor(-2..=8, -1..=1, 66);
        grid.pool(3..=5, -1..=1, 64..=65);

        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(7, 64, 0),
            &default_moves(),
            &wet_ctx(),
        )
        .unwrap();

        assert!(
            path.nodes.iter().any(|n| n.reached_by == MoveKind::Swim),
            "the channel was crossed without swimming: {:?}",
            path.nodes
                .iter()
                .map(|n| (n.pos, n.reached_by))
                .collect::<Vec<_>>()
        );
        assert!(
            path.nodes
                .iter()
                .all(|n| grid.block(n.pos) != BlockKind::Water || n.reached_by == MoveKind::Swim),
            "a water block was entered by a land move: {:?}",
            path.nodes
                .iter()
                .map(|n| (n.pos, n.reached_by))
                .collect::<Vec<_>>()
        );
        assert!(dist(path.nodes.last().unwrap().pos, BlockPos::new(7, 64, 0)) <= 1);
    }

    #[test]
    fn climbs_out_of_a_pool_onto_a_one_block_bank() {
        let mut grid = Grid::new();
        grid.floor(-2..=8, -1..=1, 63); // bank surface, stand at y=64
        grid.floor(3..=5, -1..=1, 60); // pool bottom
        grid.pool(3..=5, -1..=1, 61..=63); // surface flush with the bank top

        let path = find_path(
            &grid,
            BlockPos::new(4, 63, 0),
            BlockPos::new(7, 64, 0),
            &default_moves(),
            &wet_ctx(),
        )
        .unwrap();

        let out = path
            .nodes
            .iter()
            .find(|n| n.pos.y == 64)
            .expect("the path never left the water");
        assert_eq!(
            out.reached_by,
            MoveKind::Swim,
            "climbing out of water is a swim, not a jump: {:?}",
            path.nodes
                .iter()
                .map(|n| (n.pos, n.reached_by))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn refuses_a_pool_whose_bank_is_two_blocks_above_the_surface() {
        let mut grid = Grid::new();
        // Same pool, but the bank is two blocks proud of the water. Vanilla
        // cannot climb that from a swim, so there is no route at all.
        for y in 63..=64 {
            grid.floor(-2..=2, -1..=1, y);
            grid.floor(6..=8, -1..=1, y);
        }
        grid.floor(3..=5, -1..=1, 60);
        grid.pool(3..=5, -1..=1, 61..=63);

        let err = find_path(
            &grid,
            BlockPos::new(4, 63, 0),
            BlockPos::new(7, 65, 0),
            &default_moves(),
            &wet_ctx(),
        )
        .unwrap_err();

        assert!(matches!(err, PathError::NoPath));
    }

    #[test]
    fn prefers_a_dry_route_over_a_shorter_swim() {
        let mut grid = Grid::new();
        grid.floor(-1..=7, -1..=1, 63); // stand at y=64
        // A pond straight ahead with dry ground round its far side.
        grid.floor(3..=5, -1..=0, 60);
        grid.pool(3..=5, -1..=0, 61..=63);

        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(6, 64, 0),
            &default_moves(),
            &MoveContext {
                goal_tolerance: 0,
                ..ctx()
            },
        )
        .unwrap();

        assert!(
            path.nodes.iter().all(|n| n.reached_by != MoveKind::Swim),
            "swam through the pond despite a dry way round: {:?}",
            path.nodes
                .iter()
                .map(|n| (n.pos, n.reached_by))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_default_policy_will_not_cross_a_channel_it_could_swim() {
        // The same flooded corridor the swim test crosses. Under the default
        // policy there is no dry way round, and the correct answer is to refuse
        // rather than to plan a swim: azalea's water physics desynchronises
        // from the server every tick and gets the bot kicked.
        let mut grid = Grid::new();
        grid.floor(-2..=8, -1..=1, 63);
        grid.floor(-2..=8, -1..=1, 66);
        grid.pool(3..=5, -1..=1, 64..=65);

        let err = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(7, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .unwrap_err();

        assert!(matches!(err, PathError::NoPath));
    }

    #[test]
    fn the_default_policy_still_lets_a_bot_climb_out_of_water() {
        // Forbidding water must not strand a bot that spawned or fell into it,
        // which is exactly the state the live bots keep waking up in.
        let mut grid = Grid::new();
        grid.floor(-2..=8, -1..=1, 63); // bank surface, stand at y=64
        grid.floor(3..=5, -1..=1, 59); // pool bottom
        grid.pool(3..=5, -1..=1, 60..=63); // surface flush with the bank top

        let (path, reached) = find_path_best_effort(
            &grid,
            BlockPos::new(4, 60, 0), // submerged, three blocks down
            BlockPos::new(7, 64, 0),
            &default_moves(),
            &ctx(),
        );

        assert!(reached, "never got out of the pool: {:?}", path.nodes);
        let end = path.nodes.last().unwrap().pos;
        assert_ne!(grid.block(end), BlockKind::Water);
        // Once out, it never goes back in: the door only opens outward.
        let out_at = path
            .nodes
            .iter()
            .position(|n| grid.block(n.pos) != BlockKind::Water)
            .unwrap();
        assert!(
            path.nodes[out_at..]
                .iter()
                .all(|n| grid.block(n.pos) != BlockKind::Water),
            "re-entered forbidden water after leaving it: {:?}",
            path.nodes
                .iter()
                .map(|n| (n.pos, n.reached_by))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn best_effort_prioritizes_escaping_forbidden_water() {
        let mut grid = Grid::new();
        grid.floor(-2..=8, -1..=1, 63);
        grid.floor(3..=5, -1..=1, 59);
        grid.pool(3..=5, -1..=1, 60..=63);
        let start = BlockPos::new(4, 60, 0);
        // Swimming west is initially closer, but the goal is outside the
        // captured floor. The safe partial result must escape onto either bank.
        let goal = BlockPos::new(-20, 60, 0);

        let (path, reached) = find_path_best_effort(&grid, start, goal, &default_moves(), &ctx());

        assert!(!reached);
        assert_eq!(grid.block(start), BlockKind::Water);
        assert_ne!(
            grid.block(path.nodes.last().unwrap().pos),
            BlockKind::Water,
            "partial route stayed in forbidden water: {:?}",
            path.nodes
                .iter()
                .map(|node| (node.pos, node.reached_by))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn best_effort_returns_partial_toward_unreachable_goal() {
        // The floor ends before the goal, like an unloaded chunk boundary.
        let mut grid = Grid::new();
        grid.floor(-2..=4, -2..=2, 63);

        let start = BlockPos::new(0, 64, 0);
        let goal = BlockPos::new(20, 64, 0);
        let (path, reached) = find_path_best_effort(&grid, start, goal, &default_moves(), &ctx());

        assert!(!reached, "goal is past the floor; can't be reached yet");
        assert!(path.nodes.len() >= 2, "should still step toward the goal");
        let end = path.nodes.last().unwrap().pos;
        assert!(
            dist(end, goal) < dist(start, goal),
            "partial path {end:?} is no closer to {goal:?} than start"
        );
    }
    /// A hand-built parkour course is a chain of single blocks with nothing
    /// behind any of them, and it has to be walkable end to end.
    ///
    /// The planner used to refuse every jump on one. Any jump of three blocks
    /// or any jump that gained height needed a block behind the takeoff to run
    /// up from, and a one-block platform never has one, so a course made
    /// entirely of those jumps was invisible: a bot sent to the far end walked
    /// to the ground underneath it and reported no route. The shape here is
    /// taken from a real course - four rising three-block jumps, then a level
    /// one - because that is the shape the rule was wrong about.
    #[test]
    fn a_course_of_isolated_blocks_is_traversable() {
        let mut grid = Grid::new();
        let pillars = [(0, 36), (3, 37), (5, 38), (8, 39), (11, 40), (14, 40)];
        for (x, y) in pillars {
            grid.floor(x..=x, 0..=0, y);
        }

        let path = find_path(
            &grid,
            BlockPos::new(0, 37, 0),
            BlockPos::new(14, 41, 0),
            &default_moves(),
            &ctx(),
        )
        .expect("no route along the course");

        let end = path.nodes.last().unwrap().pos;
        assert_eq!(end, BlockPos::new(14, 41, 0), "stopped short at {end:?}");
        for (x, y) in pillars {
            let stand = BlockPos::new(x, y + 1, 0);
            assert!(
                path.nodes.iter().any(|n| n.pos == stand),
                "skipped the platform at {stand:?}"
            );
        }
    }

    /// The run-up still buys distance; it just no longer decides whether to
    /// jump at all.
    ///
    /// Four blocks of travel is the jump that genuinely needs a runway, so from
    /// an isolated block it stays refused - otherwise removing the veto would
    /// have traded a bot that never jumps for one that jumps into holes.
    #[test]
    fn the_longest_jump_still_needs_a_runway() {
        let mut lonely = Grid::new();
        lonely.floor(0..=0, 0..=0, 63);
        lonely.floor(4..=4, 0..=0, 63);
        assert!(
            find_path(
                &lonely,
                BlockPos::new(0, 64, 0),
                BlockPos::new(4, 64, 0),
                &default_moves(),
                &ctx(),
            )
            .ok()
            .is_none_or(|p| p.nodes.last().unwrap().pos != BlockPos::new(4, 64, 0)),
            "took a four-block jump with no run-up"
        );

        let mut runway = Grid::new();
        runway.floor(-3..=0, 0..=0, 63);
        runway.floor(4..=4, 0..=0, 63);
        let path = find_path(
            &runway,
            BlockPos::new(-3, 64, 0),
            BlockPos::new(4, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .expect("no route with a runway");
        assert_eq!(path.nodes.last().unwrap().pos, BlockPos::new(4, 64, 0));
    }
    /// A snowfield with an even lane and a ragged one: take the even lane.
    ///
    /// Walking beside a deeper snow layer lets a corner of the body be lifted a
    /// fraction of a block, and that is where every measured anticheat
    /// Simulation flag on this map landed. Both lanes reach the goal and the
    /// ragged one is not one block longer, so only the toll can separate them.
    #[test]
    fn a_route_prefers_the_even_side_of_a_snowfield() {
        let mut grid = Grid::new();
        // Two lanes of equal length with a junction at each end, so the only
        // thing that can decide between them is what they are made of.
        grid.floor(0..=8, -1..=-1, 63);
        grid.floor(0..=8, 1..=1, 63);
        grid.floor(0..=0, 0..=0, 63);
        grid.floor(8..=8, 0..=0, 63);
        // z = -1 is even snow all the way; z = +1 alternates depth, so every
        // block on it has a taller neighbour.
        for x in 0..=8 {
            grid.step(x, 64, -1, 4);
            grid.step(x, 64, 1, if x % 2 == 0 { 4 } else { 8 });
        }

        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(8, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .expect("no route across the field");

        let ragged = path.nodes.iter().filter(|n| n.pos.z == 1).count();
        let even = path.nodes.iter().filter(|n| n.pos.z == -1).count();
        assert!(
            even > ragged,
            "took the ragged lane: {even} even nodes vs {ragged} ragged"
        );
    }

    /// The toll must not make uneven ground impassable.
    ///
    /// A snowy mountain is built entirely of these steps, so a penalty that
    /// refuses them would trade a bot that gets flagged for one that cannot
    /// leave the hub.
    #[test]
    fn uneven_ground_is_still_crossed_when_it_is_the_only_way() {
        let mut grid = Grid::new();
        grid.floor(-1..=9, -1..=1, 63);
        for x in 0..=8 {
            grid.step(x, 64, 0, if x % 2 == 0 { 4 } else { 8 });
        }
        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(8, 64, 0),
            &default_moves(),
            &ctx(),
        )
        .expect("refused to cross uneven ground with no alternative");
        // Within the goal tolerance, which is what "reached" means here.
        let end = path.nodes.last().unwrap().pos;
        assert!(
            dist(end, BlockPos::new(8, 64, 0)) <= 1,
            "stopped at {end:?}"
        );
    }

    #[test]
    fn extreme_time_budgets_are_handled_without_panicking() {
        let mut grid = Grid::new();
        grid.floor(-2..=4, -2..=2, 63);
        let start = BlockPos::new(0, 64, 0);
        let goal = BlockPos::new(4, 64, 0);

        let zero_budget = MoveContext {
            goal_tolerance: 0,
            time_budget_ms: 0,
            ..ctx()
        };
        assert!(matches!(
            find_path(&grid, start, goal, &default_moves(), &zero_budget),
            Err(PathError::SearchBudgetExhausted)
        ));

        let unlimited_clock = MoveContext {
            goal_tolerance: 0,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        assert!(find_path(&grid, start, goal, &default_moves(), &unlimited_clock).is_ok());
    }
}
