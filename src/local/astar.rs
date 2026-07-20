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

/// Admissible A* heuristic: octile horizontal distance combined via `max()`
/// with a vertical floor. Uses `max()` (not `+`) because one move can reduce
/// horizontal AND vertical distance at once (a step-up is +1 across and +1 up),
/// so summing separate bounds double-counts and overestimates — that was the
/// old `+ 10*dy` inadmissibility. Ascending costs ≥10 per level (≥1 up-move
/// each); descending costs ≥10 per `max_fall` blocks (≥1 fall each). The true
/// path cost is ≥ both terms, so `max()` is a valid lower bound — and unlike a
/// pure-horizontal heuristic it pulls the search toward the goal's altitude
/// instead of treating every y as equal (the "ignores the y-level" bug).
fn heuristic(pos: BlockPos, goal: BlockPos, ctx: &MoveContext) -> Cost {
    let dx = (pos.x - goal.x).unsigned_abs();
    let dz = (pos.z - goal.z).unsigned_abs();
    let diagonal = ctx
        .costs
        .diagonal_walk
        .min(ctx.costs.cardinal_walk.saturating_mul(2));
    let horiz = diagonal.saturating_mul(dx.min(dz)).saturating_add(
        ctx.costs
            .cardinal_walk
            .saturating_mul(dx.max(dz) - dx.min(dz)),
    );
    let vert = if pos.y < goal.y {
        ctx.costs
            .step
            .min(ctx.costs.jump)
            .saturating_mul((goal.y - pos.y) as u32)
    } else {
        let down = (pos.y - goal.y) as u32;
        let cheapest_fall = ctx.costs.fall_base.saturating_add(ctx.costs.fall_per_block);
        let fall_bound = cheapest_fall.saturating_mul(down.div_ceil(ctx.max_fall.max(1) as u32));
        let stair_bound = ctx.costs.step.saturating_mul(down);
        fall_bound.min(stair_bound)
    };
    horiz.max(vert)
}

/// A* over block positions using the supplied move set. Succeeds when a
/// node within `ctx.goal_tolerance` (manhattan) of `goal` is expanded.
///
/// This strict "fail if unreachable" variant is the reference planner and
/// the move-rule test bed; the live executor uses [`find_path_best_effort`]
/// so it can chase goals whose chunks aren't loaded yet. Kept (rather than
/// deleted) because these tests are the movement-rule coverage.
#[allow(dead_code)]
pub fn find_path(
    world: &dyn WorldView,
    start: BlockPos,
    goal: BlockPos,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
) -> Result<Path, PathError> {
    // Admissible heuristic with vertical guidance (see `heuristic`). Wall/lava
    // penalties only ever add cost, so this stays a valid lower bound and A*
    // stays optimal.
    let h = |p: BlockPos| -> Cost { heuristic(p, goal, ctx) };
    let manhattan =
        |p: BlockPos| -> i32 { (p.x - goal.x).abs() + (p.y - goal.y).abs() + (p.z - goal.z).abs() };
    let key = |p: BlockPos| -> Key { (p.x, p.y, p.z) };

    let mut nodes: HashMap<Key, NodeData> = HashMap::new();
    // (f-score, position) — Reverse turns std's max-heap into a min-heap
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
        if manhattan(pos) <= ctx.goal_tolerance && lava_safe_to_finish(current_lava_risk, ctx) {
            return Ok(reconstruct(&nodes, pos, g));
        }

        expansions += 1;
        if expansions > ctx.max_expansions {
            return Err(PathError::SearchBudgetExhausted);
        }
        // wall-clock bound, checked sparsely to keep overhead negligible
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
            // Terrain penalties are non-negative. If the unpenalized route is
            // already no better, avoid the expensive neighbourhood scans.
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

/// Like [`find_path`], but never fails outright: when it can't reach the goal
/// (open set exhausted, or budget hit) it returns the best partial path — to
/// the reachable node closest to the goal — with `reached = false`. Walking
/// that partial path loads new chunks, so a caller that re-plans from the new
/// position can chase a goal whose chunks weren't loaded yet (viable patrol
/// waypoints beyond the initially-loaded area). `reached = true` means the
/// returned path actually reaches the goal. The path always starts at `start`;
/// if no progress is possible it's a single-node path (`nodes.len() == 1`).
pub fn find_path_best_effort(
    world: &dyn WorldView,
    start: BlockPos,
    goal: BlockPos,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
) -> (Path, bool) {
    let h = |p: BlockPos| -> Cost { heuristic(p, goal, ctx) };
    let manhattan =
        |p: BlockPos| -> i32 { (p.x - goal.x).abs() + (p.y - goal.y).abs() + (p.z - goal.z).abs() };
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
    // Closest-to-goal node reached so far, for the partial fallback. If the
    // start is unsafe, lower lava exposure ranks ahead of goal distance so a
    // partial plan leads out instead of returning the hazardous start node.
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
        if manhattan(pos) <= ctx.goal_tolerance && lava_safe_to_finish(current_lava_risk, ctx) {
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

    // couldn't reach the goal — hand back the closest reachable node so the
    // caller can walk toward the goal (loading chunks) and try again
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

    // cheap xorshift
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

    /// Hand-built grid world: solid, step, and lava blocks, else air.
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

        /// Solid floor at height `y` covering the given ranges; the bot
        /// stands at `y + 1`.
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

    #[test]
    fn tie_break_is_deterministic_bounded_and_seed_dependent() {
        let p = BlockPos::new(3, 64, -7);
        // deterministic for a given (pos, seed)
        assert_eq!(tie_break(p, 42), tie_break(p, 42));
        // a SMALL secondary key: it orders equal-f-score nodes only, and must
        // never grow large enough to override the f-score itself (which would
        // sacrifice path optimality, not just vary the route).
        for s in 0..64 {
            assert!(
                tie_break(p, s) < 3,
                "tie-break must stay a small ordering key"
            );
        }
        // the seed actually reorders positions — otherwise the path never varies
        assert!(
            (0..32).any(|s| tie_break(p, s) != tie_break(p, s + 1)),
            "the seed must vary the tie-break"
        );
    }

    fn dist(a: BlockPos, b: BlockPos) -> i32 {
        (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs()
    }

    #[test]
    fn heuristic_accounts_for_vertical_distance() {
        let goal = BlockPos::new(0, 0, 0);
        let ctx = ctx();
        // A purely-vertical offset must produce a non-zero estimate — the old
        // horizontal-only heuristic returned 0 here, so the search treated
        // "directly above/below the goal" as "already there" and ignored y.
        assert!(
            heuristic(BlockPos::new(0, 8, 0), goal, &ctx) > 0,
            "goal below ignored"
        );
        assert!(
            heuristic(BlockPos::new(0, -8, 0), goal, &ctx) > 0,
            "goal above ignored"
        );
        // Horizontal-dominant cases keep the octile value (vertical term folds
        // in via max(), never adds on top — so it can't overestimate).
        assert_eq!(heuristic(BlockPos::new(10, 0, 0), goal, &ctx), 100);
        assert_eq!(heuristic(BlockPos::new(10, 2, 0), goal, &ctx), 100);
        // A one-level descent can be a cheap stair step, so the fall-based
        // bound must never overestimate it.
        assert_eq!(
            heuristic(BlockPos::new(0, 1, 0), goal, &ctx),
            ctx.costs.step
        );
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
    fn keeps_clearance_from_walls() {
        let mut grid = Grid::new();
        grid.floor(0..=8, 1..=4, 63); // stand at y=64
        // a wall along z=0, two blocks tall
        for x in 0..=8 {
            grid.solid.insert((x, 64, 0));
            grid.solid.insert((x, 65, 0));
        }

        // start and goal hug the wall; a human drifts a block away
        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 1),
            BlockPos::new(8, 64, 1),
            &default_moves(),
            &ctx(),
        )
        .unwrap();

        let interior = &path.nodes[1..path.nodes.len() - 1];
        assert!(
            interior.iter().all(|n| n.pos.z >= 2),
            "path hugged the wall: {:?}",
            path.nodes.iter().map(|n| n.pos).collect::<Vec<_>>()
        );
    }

    /// Feet-cell has lava directly beneath it.
    fn on_lava(grid: &Grid, p: BlockPos) -> bool {
        grid.lava.contains(&(p.x, p.y - 1, p.z))
    }

    #[test]
    fn avoids_lava_when_a_dry_route_exists() {
        let mut grid = Grid::new();
        grid.floor(-1..=7, -1..=5, 63); // stand at y=64
        // A lava strip blocks the direct lane, but a route outside the default
        // two-block safety buffer remains available at z=4..5.
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
        // 1-wide corridor along z=0; the only floor cell at x=3 is lava, so
        // reaching the far side means stepping over it
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
    fn penalized_policy_can_opt_in_to_crossing_lava() {
        let mut grid = Grid::new();
        grid.floor(-1..=7, 0..=0, 63);
        grid.solid.remove(&(3, 63, 0));
        grid.lava.insert((3, 63, 0));
        let mut ctx = ctx();
        ctx.lava_policy = super::super::moves::LavaPolicy::Penalized;

        let path = find_path(
            &grid,
            BlockPos::new(0, 64, 0),
            BlockPos::new(6, 64, 0),
            &default_moves(),
            &ctx,
        )
        .expect("the explicit legacy policy should permit the only route");

        assert!(path.nodes.iter().any(|n| on_lava(&grid, n.pos)));
    }

    #[test]
    fn best_effort_prioritizes_escaping_an_existing_lava_hazard() {
        let mut grid = Grid::new();
        // The only exit is east, while the unreachable goal is west. A
        // distance-only partial fallback would remain on lava because every
        // safe step initially moves farther from the goal.
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
        // seal a 5x5 room: walls all around, lid on top
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
    fn best_effort_reaches_a_reachable_goal() {
        let mut grid = Grid::new();
        grid.floor(-2..=8, -2..=2, 63);

        let goal = BlockPos::new(5, 64, 0);
        let (path, reached) = find_path_best_effort(
            &grid,
            BlockPos::new(0, 64, 0),
            goal,
            &default_moves(),
            &ctx(),
        );

        assert!(reached, "should reach a goal on open floor");
        assert!(dist(path.nodes.last().unwrap().pos, goal) <= 1);
    }

    #[test]
    fn best_effort_returns_partial_toward_unreachable_goal() {
        // floor only reaches x=4; the goal at x=20 sits past the known world
        // (like an unloaded chunk). Best-effort should walk toward it, not fail.
        let mut grid = Grid::new();
        grid.floor(-2..=4, -2..=2, 63);

        let start = BlockPos::new(0, 64, 0);
        let goal = BlockPos::new(20, 64, 0);
        let (path, reached) = find_path_best_effort(&grid, start, goal, &default_moves(), &ctx());

        assert!(!reached, "goal is past the floor; can't be reached yet");
        assert!(path.nodes.len() >= 2, "should still step toward the goal");
        // the partial path must end closer to the goal than the start
        let end = path.nodes.last().unwrap().pos;
        assert!(
            dist(end, goal) < dist(start, goal),
            "partial path {end:?} is no closer to {goal:?} than start"
        );
    }
}
