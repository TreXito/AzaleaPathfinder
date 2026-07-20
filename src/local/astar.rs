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
}

struct NodeData {
    g: Cost,
    parent: Option<BlockPos>,
    reached_by: MoveKind,
}

/// Lower bound for A*: octile horizontal cost and vertical cost are combined
/// with `max` because one move can advance on both axes.
fn heuristic(pos: BlockPos, goal: BlockPos, ctx: &MoveContext) -> Cost {
    // Applying tolerance to each axis is looser than Manhattan tolerance, so
    // the estimate stays admissible.
    let tolerance = ctx.goal_tolerance.max(0) as u32;
    let dx = pos.x.abs_diff(goal.x).saturating_sub(tolerance);
    let dz = pos.z.abs_diff(goal.z).saturating_sub(tolerance);
    let dy = pos.y.abs_diff(goal.y).saturating_sub(tolerance);
    // Jump, step, and fall can also make horizontal progress.
    let cardinal = ctx
        .costs
        .cardinal_walk
        .min(ctx.costs.step)
        .min(ctx.costs.jump)
        .min(ctx.costs.fall_base.saturating_add(ctx.costs.fall_per_block));
    let diagonal = ctx.costs.diagonal_walk.min(cardinal.saturating_mul(2));
    let horiz = diagonal
        .saturating_mul(dx.min(dz))
        .saturating_add(cardinal.saturating_mul(dx.max(dz) - dx.min(dz)));
    let vert = if pos.y < goal.y {
        ctx.costs.step.min(ctx.costs.jump).saturating_mul(dy)
    } else {
        let cheapest_fall = ctx.costs.fall_base.saturating_add(ctx.costs.fall_per_block);
        let fall_bound = cheapest_fall.saturating_mul(dy.div_ceil(ctx.max_fall.max(1) as u32));
        let stair_bound = ctx.costs.step.saturating_mul(dy);
        fall_bound.min(stair_bound)
    };
    horiz.max(vert)
}

/// Finds a path to within `ctx.goal_tolerance` Manhattan distance of `goal`.
/// Returns an error instead of a partial path when the goal cannot be reached.
#[allow(dead_code)]
pub fn find_path(
    world: &dyn WorldView,
    start: BlockPos,
    goal: BlockPos,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
) -> Result<Path, PathError> {
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
    // Reverse turns the max-heap into a min-heap ordered by f-score.
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

    let mut expansions = 0usize;
    let mut scratch: Vec<Edge> = Vec::new();
    let mut terrain_costs: HashMap<Key, Cost> = HashMap::new();
    let mut lava_risks: HashMap<Key, Cost> = HashMap::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(ctx.time_budget_ms);

    while let Some(Reverse((f, _, k))) = open.pop() {
        let pos = BlockPos::new(k.0, k.1, k.2);
        let g = nodes[&k].g;
        if f > g.saturating_add(h(pos)) {
            continue; // stale heap entry superseded by a cheaper route
        }

        let current_lava_risk = cached_lava_risk(pos, world, ctx, &mut lava_risks);
        if manhattan(pos) <= goal_tolerance && lava_safe_to_finish(current_lava_risk, ctx) {
            return Ok(reconstruct(&nodes, pos, g));
        }

        expansions += 1;
        if expansions > ctx.max_expansions {
            return Err(PathError::SearchBudgetExhausted);
        }
        // Check time occasionally to avoid paying for a clock read per node.
        if expansions.is_multiple_of(512) && std::time::Instant::now() > deadline {
            return Err(PathError::SearchBudgetExhausted);
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
            // Skip the terrain scan if the base cost cannot improve this node.
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

    Err(PathError::NoPath)
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
    // When starting near lava, escaping it takes priority over goal distance.
    let mut best_risk = cached_lava_risk(start, world, ctx, &mut lava_risks);
    let mut best_dist = manhattan(start);
    let mut best = (start, 0u32);

    let mut expansions = 0usize;
    let mut scratch: Vec<Edge> = Vec::new();
    let mut terrain_costs: HashMap<Key, Cost> = HashMap::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(ctx.time_budget_ms);

    while let Some(Reverse((f, _, k))) = open.pop() {
        let pos = BlockPos::new(k.0, k.1, k.2);
        let g = nodes[&k].g;
        if f > g.saturating_add(h(pos)) {
            continue;
        }
        let current_lava_risk = cached_lava_risk(pos, world, ctx, &mut lava_risks);
        if manhattan(pos) <= goal_tolerance && lava_safe_to_finish(current_lava_risk, ctx) {
            return (reconstruct(&nodes, pos, g), true);
        }
        let md = manhattan(pos);
        if current_lava_risk < best_risk || (current_lava_risk == best_risk && md < best_dist) {
            best_risk = current_lava_risk;
            best_dist = md;
            best = (pos, g);
        }
        expansions += 1;
        if expansions > ctx.max_expansions {
            break;
        }
        if expansions.is_multiple_of(512) && std::time::Instant::now() > deadline {
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

    // A partial path lets the caller move, load more chunks, and plan again.
    (reconstruct(&nodes, best.0, best.1), false)
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
        steps: HashSet<(i32, i32, i32)>,
        lava: HashSet<(i32, i32, i32)>,
    }

    impl Grid {
        fn new() -> Self {
            Self {
                solid: HashSet::new(),
                steps: HashSet::new(),
                lava: HashSet::new(),
            }
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
            } else if self.steps.contains(&k) {
                BlockKind::Step
            } else if self.lava.contains(&k) {
                BlockKind::Lava
            } else {
                BlockKind::Air
            }
        }
    }

    fn ctx() -> MoveContext {
        MoveContext::default()
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
        grid.steps.insert((3, 64, 0)); // stair
        grid.steps.insert((4, 65, 0)); // stair
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
}
