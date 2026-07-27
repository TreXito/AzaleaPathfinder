use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant};

use azalea::BlockPos;

use super::moves::{Edge, Move, MoveContext};
use super::world::WorldView;
use crate::adaptive::CostModelSnapshot;
use crate::planning::{MotionMetadata, MotionPlan, MotionPlanError, MotionStep};
use crate::types::{Cost, MoveKind, Path, PathError, PathNode};

type Key = (i32, i32, i32);
pub const MAX_CANDIDATES_PER_MOVE: usize = 4_096;

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

#[derive(Clone, Copy)]
struct NodeData {
    g: Cost,
    parent: Option<BlockPos>,
    reached_by: MoveKind,
    metadata: Option<MotionMetadata>,
}

/// Lower bound for A*. Horizontal and vertical estimates are combined with
/// `max` because one move can advance on both axes.
fn block_heuristic(pos: BlockPos, goal: BlockPos, tolerance: i32, ctx: &MoveContext) -> Cost {
    // Applying tolerance to each axis is looser than Manhattan tolerance, so
    // the estimate stays admissible.
    let tolerance = tolerance.max(0) as u32;
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

/// A destination predicate for a path search.
///
/// Goals are deliberately separate from movement rules. A goal may supply an
/// admissible lower bound, but returning `None` is always correct and makes the
/// search use Dijkstra ordering. Implementations that return `Some` must ensure
/// the value never exceeds the cheapest remaining path cost and is zero for
/// every position accepted by [`Goal::is_satisfied`].
///
/// [`Goal::progress_key`] is used only to choose an anytime/best-effort partial
/// result. It cannot affect the optimality of a completed path.
///
/// Every method result must remain stable while [`Goal::revision`] is stable.
pub trait Goal: Send + Sync {
    /// Whether `pos` completes this search.
    fn is_satisfied(&self, pos: BlockPos) -> bool;

    /// A guaranteed lower bound on the cost from `pos` to this goal.
    ///
    /// The planner also disables this value whenever any movement rule opts
    /// out of the built-in heuristic assumptions.
    fn heuristic_lower_bound(&self, _pos: BlockPos, _ctx: &MoveContext) -> Option<Cost> {
        None
    }

    /// Smaller values represent better partial progress.
    fn progress_key(&self, _pos: BlockPos) -> u64 {
        u64::MAX
    }

    /// Monotonic identity/version for goals backed by changing state.
    ///
    /// A changed value invalidates a resumed session before it does more work.
    fn revision(&self) -> u64 {
        0
    }
}

/// The legacy "within Manhattan tolerance of one block" destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGoal {
    target: BlockPos,
    tolerance: i32,
    revision: u64,
}

impl BlockGoal {
    pub fn new(target: BlockPos, tolerance: i32) -> Self {
        Self {
            target,
            tolerance: tolerance.max(0),
            revision: 0,
        }
    }

    /// Attaches an application-owned revision to this immutable goal value.
    pub fn with_revision(mut self, revision: u64) -> Self {
        self.revision = revision;
        self
    }

    pub fn target(&self) -> BlockPos {
        self.target
    }

    pub fn tolerance(&self) -> i32 {
        self.tolerance
    }

    fn manhattan(&self, pos: BlockPos) -> u64 {
        u64::from(pos.x.abs_diff(self.target.x))
            .saturating_add(u64::from(pos.y.abs_diff(self.target.y)))
            .saturating_add(u64::from(pos.z.abs_diff(self.target.z)))
    }
}

impl Goal for BlockGoal {
    fn is_satisfied(&self, pos: BlockPos) -> bool {
        self.manhattan(pos) <= u64::from(self.tolerance as u32)
    }

    fn heuristic_lower_bound(&self, pos: BlockPos, ctx: &MoveContext) -> Option<Cost> {
        Some(block_heuristic(pos, self.target, self.tolerance, ctx))
    }

    fn progress_key(&self, pos: BlockPos) -> u64 {
        self.manhattan(pos)
    }

    fn revision(&self) -> u64 {
        self.revision
    }
}

/// Externally maintained revisions frozen into a resumable search.
///
/// `MoveContext` itself is immutably borrowed for the session lifetime. These
/// numbers cover state not represented by that borrow, such as a mutable world
/// snapshot generation or a newly promoted adaptive cost model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchAssumptions {
    pub world_revision: u64,
    pub cost_revision: u64,
}

/// The part of a frozen search assumption that changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionKind {
    World,
    Cost,
    Goal,
}

/// Why a resumed search was rejected without changing its frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevisionMismatch {
    pub kind: RevisionKind,
    pub expected: u64,
    pub actual: u64,
}

/// A cooperative amount of work for [`SearchSession::advance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchSlice {
    /// Maximum non-stale nodes whose outgoing edges may be generated.
    pub max_expansions: usize,
    /// Optional wall-clock limit for this call. This is independent of the
    /// cumulative hard limit in [`MoveContext::time_budget_ms`].
    pub time_budget: Option<Duration>,
}

impl SearchSlice {
    pub const fn new(max_expansions: usize) -> Self {
        Self {
            max_expansions,
            time_budget: None,
        }
    }

    pub const fn with_time_budget(mut self, time_budget: Duration) -> Self {
        self.time_budget = Some(time_budget);
        self
    }

    pub const fn unlimited() -> Self {
        Self::new(usize::MAX)
    }
}

impl Default for SearchSlice {
    fn default() -> Self {
        Self::new(2_048).with_time_budget(Duration::from_millis(5))
    }
}

/// Result of advancing a resumable search.
#[derive(Debug, Clone)]
pub enum SearchStatus {
    /// The frontier and all caches remain available for another slice.
    InProgress,
    /// An optimal path was popped from the frontier.
    Found(Path),
    /// Every reachable node was considered without satisfying the goal.
    Exhausted,
    /// The cumulative expansion or wall-clock budget was reached.
    BudgetExhausted,
    /// A goal or checked external revision changed. No search work was done.
    Invalidated(RevisionMismatch),
}

#[derive(Debug, Clone)]
enum TerminalStatus {
    Found(Path),
    Exhausted,
    BudgetExhausted,
}

impl TerminalStatus {
    fn public(&self) -> SearchStatus {
        match self {
            Self::Found(path) => SearchStatus::Found(path.clone()),
            Self::Exhausted => SearchStatus::Exhausted,
            Self::BudgetExhausted => SearchStatus::BudgetExhausted,
        }
    }
}

/// Stateful A*/Dijkstra search that can be advanced in bounded slices.
///
/// The world, movement registry, goal and movement context are borrowed for
/// the full lifetime of the session, freezing the cost/validation assumptions
/// represented by those values. Use [`SearchSession::advance_checked`] when
/// the world snapshot or adaptive model has an application-owned revision.
/// Movement rules with interior mutability must likewise keep their candidate
/// generation stable for the session lifetime.
pub struct SearchSession<'a> {
    world: &'a dyn WorldView,
    moves: &'a [Box<dyn Move>],
    ctx: &'a MoveContext,
    cost_model: Option<&'a CostModelSnapshot>,
    goal: &'a dyn Goal,
    goal_revision: u64,
    assumptions: SearchAssumptions,
    use_heuristic: bool,
    nodes: HashMap<Key, NodeData>,
    // (f, deterministic tie break, position, g at insertion). Recording g
    // makes stale entries unambiguous even when f saturates at Cost::MAX.
    open: BinaryHeap<Reverse<(Cost, Cost, Key, Cost)>>,
    lava_risks: HashMap<Key, Cost>,
    terrain_costs: HashMap<Key, Cost>,
    scratch: Vec<Edge>,
    best_lava_risk: Cost,
    best_water_risk: u8,
    best_progress: u64,
    best: (BlockPos, Cost),
    expansions: usize,
    hard_time_budget: Duration,
    elapsed_compute: Duration,
    terminal: Option<TerminalStatus>,
}

impl<'a> SearchSession<'a> {
    pub fn new(
        world: &'a dyn WorldView,
        start: BlockPos,
        goal: &'a dyn Goal,
        moves: &'a [Box<dyn Move>],
        ctx: &'a MoveContext,
    ) -> Self {
        Self::new_with_assumptions_and_cost_model(
            world,
            start,
            goal,
            moves,
            ctx,
            SearchAssumptions::default(),
            None,
        )
    }

    pub fn new_with_cost_model(
        world: &'a dyn WorldView,
        start: BlockPos,
        goal: &'a dyn Goal,
        moves: &'a [Box<dyn Move>],
        ctx: &'a MoveContext,
        cost_model: &'a CostModelSnapshot,
    ) -> Self {
        Self::new_with_assumptions_and_cost_model(
            world,
            start,
            goal,
            moves,
            ctx,
            SearchAssumptions {
                world_revision: 0,
                cost_revision: cost_model.model_revision,
            },
            Some(cost_model),
        )
    }

    pub fn new_with_assumptions(
        world: &'a dyn WorldView,
        start: BlockPos,
        goal: &'a dyn Goal,
        moves: &'a [Box<dyn Move>],
        ctx: &'a MoveContext,
        assumptions: SearchAssumptions,
    ) -> Self {
        Self::new_with_assumptions_and_cost_model(world, start, goal, moves, ctx, assumptions, None)
    }

    pub fn new_with_assumptions_and_cost_model(
        world: &'a dyn WorldView,
        start: BlockPos,
        goal: &'a dyn Goal,
        moves: &'a [Box<dyn Move>],
        ctx: &'a MoveContext,
        assumptions: SearchAssumptions,
        cost_model: Option<&'a CostModelSnapshot>,
    ) -> Self {
        let use_heuristic = cost_model.is_none_or(|model| !model.affects_routing())
            && moves
                .iter()
                .all(|movement| movement.supports_builtin_heuristic())
            && goal.heuristic_lower_bound(start, ctx).is_some();
        let mut nodes = HashMap::new();
        nodes.insert(
            key(start),
            NodeData {
                g: 0,
                parent: None,
                reached_by: MoveKind::Start,
                metadata: None,
            },
        );
        let initial_h = if use_heuristic {
            goal.heuristic_lower_bound(start, ctx).unwrap_or(0)
        } else {
            0
        };
        let mut open = BinaryHeap::new();
        open.push(Reverse((
            initial_h,
            tie_break(start, ctx.path_seed),
            key(start),
            0,
        )));

        let mut lava_risks = HashMap::new();
        let best_lava_risk = cached_lava_risk(start, world, ctx, &mut lava_risks);
        let best_water_risk = water_risk(start, world, ctx);
        let best_progress = goal.progress_key(start);
        Self {
            world,
            moves,
            ctx,
            cost_model,
            goal,
            goal_revision: goal.revision(),
            assumptions,
            use_heuristic,
            nodes,
            open,
            lava_risks,
            terrain_costs: HashMap::new(),
            scratch: Vec::new(),
            best_lava_risk,
            best_water_risk,
            best_progress,
            best: (start, 0),
            expansions: 0,
            hard_time_budget: Duration::from_millis(ctx.time_budget_ms),
            elapsed_compute: Duration::ZERO,
            terminal: None,
        }
    }

    pub fn assumptions(&self) -> SearchAssumptions {
        self.assumptions
    }

    pub fn expansions(&self) -> usize {
        self.expansions
    }

    /// Wall-clock time spent inside positive [`SearchSession::advance`] calls.
    ///
    /// Time between slices is deliberately excluded, so a cooperatively
    /// scheduled session does not expire merely because it yielded.
    pub fn elapsed_compute(&self) -> Duration {
        self.elapsed_compute
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    /// Best safe progress known so far. This is always at least `[start]`.
    pub fn best_path(&self) -> Path {
        reconstruct(&self.nodes, self.best.0, self.best.1)
    }

    pub fn motion_plan_for(&self, path: Path) -> Result<MotionPlan, MotionPlanError> {
        let first = path
            .nodes
            .first()
            .ok_or(MotionPlanError::MetadataMismatch)?;
        let first_data = self
            .nodes
            .get(&key(first.pos))
            .ok_or(MotionPlanError::MissingGeneratedMetadata)?;
        if first.reached_by != MoveKind::Start
            || first_data.parent.is_some()
            || first_data.reached_by != MoveKind::Start
        {
            return Err(MotionPlanError::MetadataMismatch);
        }
        let end_data = self
            .nodes
            .get(&key(path
                .nodes
                .last()
                .expect("non-empty path checked above")
                .pos))
            .ok_or(MotionPlanError::MissingGeneratedMetadata)?;
        if end_data.g != path.total_cost {
            return Err(MotionPlanError::MetadataMismatch);
        }

        let mut steps = Vec::with_capacity(path.nodes.len().saturating_sub(1));
        for (edge_index, nodes) in path.nodes.windows(2).enumerate() {
            let node = self
                .nodes
                .get(&key(nodes[1].pos))
                .ok_or(MotionPlanError::MissingGeneratedMetadata)?;
            if node.parent != Some(nodes[0].pos) || node.reached_by != nodes[1].reached_by {
                return Err(MotionPlanError::MetadataMismatch);
            }
            let metadata = node
                .metadata
                .ok_or(MotionPlanError::MissingGeneratedMetadata)?;
            steps.push(MotionStep {
                edge_index: edge_index as u32,
                from: nodes[0].pos,
                to: nodes[1].pos,
                kind: nodes[1].reached_by,
                primitive: metadata.primitive,
                features: metadata.features,
                predicted: metadata.predicted,
                predicted_ticks: metadata.predicted_ticks,
            });
        }
        MotionPlan::from_parts(path, steps)
    }

    /// Advances only if caller-maintained world and cost revisions still match.
    pub fn advance_checked(
        &mut self,
        slice: SearchSlice,
        current: SearchAssumptions,
    ) -> SearchStatus {
        if current.world_revision != self.assumptions.world_revision {
            return SearchStatus::Invalidated(RevisionMismatch {
                kind: RevisionKind::World,
                expected: self.assumptions.world_revision,
                actual: current.world_revision,
            });
        }
        if current.cost_revision != self.assumptions.cost_revision {
            return SearchStatus::Invalidated(RevisionMismatch {
                kind: RevisionKind::Cost,
                expected: self.assumptions.cost_revision,
                actual: current.cost_revision,
            });
        }
        self.advance(slice)
    }

    /// Performs at most `slice.max_expansions` units of search work.
    ///
    /// A zero-expansion slice is strictly observational: it does not pop stale
    /// entries, inspect mutable goal state, start timers or change terminal
    /// state.
    pub fn advance(&mut self, slice: SearchSlice) -> SearchStatus {
        if slice.max_expansions == 0 {
            return self
                .terminal
                .as_ref()
                .map_or(SearchStatus::InProgress, TerminalStatus::public);
        }
        let current_goal_revision = self.goal.revision();
        if current_goal_revision != self.goal_revision {
            return SearchStatus::Invalidated(RevisionMismatch {
                kind: RevisionKind::Goal,
                expected: self.goal_revision,
                actual: current_goal_revision,
            });
        }
        if let Some(terminal) = &self.terminal {
            return terminal.public();
        }

        let call_started = Instant::now();
        let mut slice_expansions = 0usize;

        loop {
            if self.hard_time_exhausted(call_started) {
                self.terminal = Some(TerminalStatus::BudgetExhausted);
                return self.finish_advance(call_started, SearchStatus::BudgetExhausted);
            }
            if slice_expansions >= slice.max_expansions
                || slice
                    .time_budget
                    .is_some_and(|budget| call_started.elapsed() >= budget)
            {
                return self.finish_advance(call_started, SearchStatus::InProgress);
            }

            let Some(Reverse((_f, _tie, k, queued_g))) = self.open.pop() else {
                self.terminal = Some(TerminalStatus::Exhausted);
                return self.finish_advance(call_started, SearchStatus::Exhausted);
            };
            let Some(node) = self.nodes.get(&k).copied() else {
                continue;
            };
            // Reopening inserts a new entry rather than mutating the heap.
            if queued_g != node.g {
                // Stale heap cleanup is not an expansion, but it still must
                // not let an unlimited slice run past the cumulative clock.
                if self.hard_time_exhausted(call_started) {
                    self.terminal = Some(TerminalStatus::BudgetExhausted);
                    return self.finish_advance(call_started, SearchStatus::BudgetExhausted);
                }
                continue;
            }

            let pos = BlockPos::new(k.0, k.1, k.2);
            let g = node.g;
            let current_lava_risk =
                cached_lava_risk(pos, self.world, self.ctx, &mut self.lava_risks);
            let current_water_risk = water_risk(pos, self.world, self.ctx);
            if self.goal.is_satisfied(pos)
                && lava_safe_to_finish(current_lava_risk, self.ctx)
                && current_water_risk == 0
            {
                let path = reconstruct(&self.nodes, pos, g);
                self.terminal = Some(TerminalStatus::Found(path.clone()));
                return self.finish_advance(call_started, SearchStatus::Found(path));
            }

            let progress = self.goal.progress_key(pos);
            if current_lava_risk < self.best_lava_risk
                || (current_lava_risk == self.best_lava_risk
                    && (current_water_risk < self.best_water_risk
                        || (current_water_risk == self.best_water_risk
                            && progress < self.best_progress)))
            {
                self.best_lava_risk = current_lava_risk;
                self.best_water_risk = current_water_risk;
                self.best_progress = progress;
                self.best = (pos, g);
            }

            // Goal testing deliberately happens before budget testing. Popping
            // a goal does not expand it and a goal discovered by the final
            // permitted expansion must still be returned.
            if self.expansions >= self.ctx.max_expansions || self.hard_time_exhausted(call_started)
            {
                self.terminal = Some(TerminalStatus::BudgetExhausted);
                return self.finish_advance(call_started, SearchStatus::BudgetExhausted);
            }

            self.expansions = self.expansions.saturating_add(1);
            slice_expansions = slice_expansions.saturating_add(1);
            let mut scratch = std::mem::take(&mut self.scratch);
            for movement in self.moves {
                scratch.clear();
                movement.candidates(pos, self.world, self.ctx, &mut scratch);
                if scratch.len() > MAX_CANDIDATES_PER_MOVE {
                    self.scratch = Vec::new();
                    self.terminal = Some(TerminalStatus::BudgetExhausted);
                    return self.finish_advance(call_started, SearchStatus::BudgetExhausted);
                }
                for edge in scratch.drain(..) {
                    if !lava_edge_allowed(pos, edge.to, self.world, self.ctx, &mut self.lava_risks)
                    {
                        continue;
                    }
                    let metadata = movement
                        .metadata(pos, &edge, self.world, self.ctx)
                        .filter(|metadata| metadata.predicted.total() == edge.cost);
                    let adaptable_cost = match (self.cost_model, metadata) {
                        (Some(model), Some(metadata)) => model
                            .edge_components(
                                metadata.primitive,
                                metadata.features,
                                metadata.predicted,
                            )
                            .total(),
                        _ => edge.cost,
                    };
                    let edge_key = key(edge.to);
                    let base_g = g.saturating_add(adaptable_cost);
                    if self
                        .nodes
                        .get(&edge_key)
                        .is_some_and(|known| base_g >= known.g)
                    {
                        continue;
                    }
                    let terrain = *self
                        .terrain_costs
                        .entry(edge_key)
                        .or_insert_with(|| terrain_penalty(edge.to, self.world, self.ctx));
                    let next_g = base_g.saturating_add(terrain);
                    if self
                        .nodes
                        .get(&edge_key)
                        .is_some_and(|known| next_g >= known.g)
                    {
                        continue;
                    }

                    self.nodes.insert(
                        edge_key,
                        NodeData {
                            g: next_g,
                            parent: Some(pos),
                            reached_by: edge.kind,
                            metadata,
                        },
                    );
                    let next_h = self.heuristic(edge.to);
                    self.open.push(Reverse((
                        next_g.saturating_add(next_h),
                        tie_break(edge.to, self.ctx.path_seed),
                        edge_key,
                        next_g,
                    )));
                }
            }
            self.scratch = scratch;
        }
    }

    fn heuristic(&self, pos: BlockPos) -> Cost {
        if self.use_heuristic {
            self.goal.heuristic_lower_bound(pos, self.ctx).unwrap_or(0)
        } else {
            0
        }
    }

    fn hard_time_exhausted(&self, call_started: Instant) -> bool {
        self.elapsed_compute.saturating_add(call_started.elapsed()) >= self.hard_time_budget
    }

    fn finish_advance(&mut self, call_started: Instant, status: SearchStatus) -> SearchStatus {
        self.elapsed_compute = self.elapsed_compute.saturating_add(call_started.elapsed());
        status
    }
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
    let block_goal = BlockGoal::new(goal, ctx.goal_tolerance);
    find_path_to_goal(world, start, &block_goal, moves, ctx)
}

/// Finds an optimal path satisfying an object-safe [`Goal`].
pub fn find_path_to_goal(
    world: &dyn WorldView,
    start: BlockPos,
    goal: &dyn Goal,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
) -> Result<Path, PathError> {
    let mut session = SearchSession::new(world, start, goal, moves, ctx);
    loop {
        match session.advance(SearchSlice::unlimited()) {
            SearchStatus::InProgress => continue,
            SearchStatus::Found(path) => return Ok(path),
            SearchStatus::Exhausted => return Err(PathError::NoPath),
            SearchStatus::BudgetExhausted => return Err(PathError::SearchBudgetExhausted),
            SearchStatus::Invalidated(_) => return Err(PathError::Cancelled),
        }
    }
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
    let block_goal = BlockGoal::new(goal, ctx.goal_tolerance);
    find_path_to_goal_best_effort(world, start, &block_goal, moves, ctx)
}

/// Goal-trait counterpart to [`find_path_best_effort`].
///
/// A concurrent goal revision change stops the search and returns the best
/// partial path with `false`; use [`SearchSession`] directly when the caller
/// needs to distinguish invalidation from exhaustion.
pub fn find_path_to_goal_best_effort(
    world: &dyn WorldView,
    start: BlockPos,
    goal: &dyn Goal,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
) -> (Path, bool) {
    let mut session = SearchSession::new(world, start, goal, moves, ctx);
    loop {
        match session.advance(SearchSlice::unlimited()) {
            SearchStatus::InProgress => continue,
            SearchStatus::Found(path) => return (path, true),
            SearchStatus::Exhausted | SearchStatus::BudgetExhausted => {
                return (session.best_path(), false);
            }
            SearchStatus::Invalidated(_) => return (session.best_path(), false),
        }
    }
}

/// Plans with generator-authored motion metadata and an optional frozen
/// adaptive model. Missing metadata from a custom movement rule is reported
/// rather than reverse-inferred.
pub fn find_motion_plan_best_effort(
    world: &dyn WorldView,
    start: BlockPos,
    goal: BlockPos,
    moves: &[Box<dyn Move>],
    ctx: &MoveContext,
    cost_model: Option<&CostModelSnapshot>,
) -> Result<(MotionPlan, bool), MotionPlanError> {
    let block_goal = BlockGoal::new(goal, ctx.goal_tolerance);
    let mut session = match cost_model {
        Some(model) => {
            SearchSession::new_with_cost_model(world, start, &block_goal, moves, ctx, model)
        }
        None => SearchSession::new(world, start, &block_goal, moves, ctx),
    };
    loop {
        match session.advance(SearchSlice::unlimited()) {
            SearchStatus::InProgress => {}
            SearchStatus::Found(path) => {
                return session.motion_plan_for(path).map(|plan| (plan, true));
            }
            SearchStatus::Exhausted
            | SearchStatus::BudgetExhausted
            | SearchStatus::Invalidated(_) => {
                return session
                    .motion_plan_for(session.best_path())
                    .map(|plan| (plan, false));
            }
        }
    }
}

fn key(pos: BlockPos) -> Key {
    (pos.x, pos.y, pos.z)
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

    fn graph_node(id: i32) -> BlockPos {
        BlockPos::new(id, 64, 0)
    }

    struct GraphMove {
        edges: std::collections::HashMap<i32, Vec<(i32, Cost)>>,
        heuristic_compatible: bool,
    }

    impl GraphMove {
        fn new(edges: &[(i32, i32, Cost)], heuristic_compatible: bool) -> Self {
            let mut by_source: std::collections::HashMap<i32, Vec<(i32, Cost)>> =
                std::collections::HashMap::new();
            for &(from, to, cost) in edges {
                by_source.entry(from).or_default().push((to, cost));
            }
            Self {
                edges: by_source,
                heuristic_compatible,
            }
        }
    }

    impl Move for GraphMove {
        fn candidates(
            &self,
            from: BlockPos,
            _world: &dyn WorldView,
            _ctx: &MoveContext,
            out: &mut Vec<Edge>,
        ) {
            for &(to, cost) in self.edges.get(&from.x).into_iter().flatten() {
                out.push(Edge {
                    to: graph_node(to),
                    kind: MoveKind::Walk,
                    cost,
                });
            }
        }

        fn supports_builtin_heuristic(&self) -> bool {
            self.heuristic_compatible
        }

        fn metadata(
            &self,
            from: BlockPos,
            edge: &Edge,
            _world: &dyn WorldView,
            _ctx: &MoveContext,
        ) -> Option<MotionMetadata> {
            Some(MotionMetadata {
                primitive: crate::adaptive::PrimitiveId::WALK_CARDINAL,
                features: crate::adaptive::MoveFeatures::new(
                    from.x.abs_diff(edge.to.x).min(16) as u8,
                    0,
                    crate::adaptive::TerrainClass::Unknown,
                ),
                predicted: crate::adaptive::CostComponents {
                    time: edge.cost,
                    ..crate::adaptive::CostComponents::default()
                },
                predicted_ticks: 1,
            })
        }
    }

    struct GraphGoal {
        target: i32,
        lower_bounds: std::collections::HashMap<i32, Cost>,
    }

    impl Goal for GraphGoal {
        fn is_satisfied(&self, pos: BlockPos) -> bool {
            pos.x == self.target
        }

        fn heuristic_lower_bound(&self, pos: BlockPos, _ctx: &MoveContext) -> Option<Cost> {
            Some(self.lower_bounds.get(&pos.x).copied().unwrap_or(0))
        }

        fn progress_key(&self, pos: BlockPos) -> u64 {
            u64::from(pos.x.abs_diff(self.target))
        }
    }

    #[test]
    fn one_node_slices_are_equivalent_to_one_shot_search() {
        let mut grid = Grid::new();
        grid.floor(-2..=10, -2..=2, 63);
        let start = BlockPos::new(0, 64, 0);
        let destination = BlockPos::new(8, 64, 0);
        let context = MoveContext {
            goal_tolerance: 0,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let moves = default_moves();
        let expected = find_path(&grid, start, destination, &moves, &context).unwrap();
        let goal = BlockGoal::new(destination, 0);
        let mut session = SearchSession::new(&grid, start, &goal, &moves, &context);

        let actual = loop {
            match session.advance(SearchSlice::new(1)) {
                SearchStatus::InProgress => {}
                SearchStatus::Found(path) => break path,
                status => panic!("tiny-slice search ended unexpectedly: {status:?}"),
            }
        };

        assert_eq!(actual.total_cost, expected.total_cost);
        assert_eq!(
            actual.nodes.iter().map(|node| node.pos).collect::<Vec<_>>(),
            expected
                .nodes
                .iter()
                .map(|node| node.pos)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .nodes
                .iter()
                .map(|node| node.reached_by)
                .collect::<Vec<_>>(),
            expected
                .nodes
                .iter()
                .map(|node| node.reached_by)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn zero_slice_is_side_effect_free_even_for_a_completed_start() {
        let grid = Grid::new();
        let context = MoveContext {
            goal_tolerance: 0,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let moves: Vec<Box<dyn Move>> = Vec::new();
        let start = graph_node(3);
        let goal = BlockGoal::new(start, 0);
        let mut session = SearchSession::new(&grid, start, &goal, &moves, &context);
        let before = session.best_path();

        assert!(matches!(
            session.advance(SearchSlice::new(0)),
            SearchStatus::InProgress
        ));
        assert_eq!(session.expansions(), 0);
        assert_eq!(session.elapsed_compute(), Duration::ZERO);
        assert!(!session.is_terminal());
        assert_eq!(session.best_path().total_cost, before.total_cost);
        assert_eq!(session.best_path().nodes[0].pos, before.nodes[0].pos);

        assert!(matches!(
            session.advance(SearchSlice::new(1)),
            SearchStatus::Found(_)
        ));
        assert_eq!(session.expansions(), 0, "a goal node is never expanded");
    }

    #[test]
    fn per_call_slice_limit_does_not_reset_the_cumulative_hard_budget() {
        let grid = Grid::new();
        let context = MoveContext {
            goal_tolerance: 0,
            max_expansions: 2,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let moves: Vec<Box<dyn Move>> = vec![Box::new(GraphMove::new(
            &[(0, 1, 1), (1, 2, 1), (2, 3, 1)],
            false,
        ))];
        let goal = BlockGoal::new(graph_node(3), 0);
        let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);

        assert!(matches!(
            session.advance(SearchSlice::new(1)),
            SearchStatus::InProgress
        ));
        assert!(matches!(
            session.advance(SearchSlice::new(1)),
            SearchStatus::InProgress
        ));
        assert_eq!(session.expansions(), 2);
        assert!(matches!(
            session.advance(SearchSlice::new(1)),
            SearchStatus::BudgetExhausted
        ));
        assert_eq!(session.expansions(), 2);
        assert_eq!(session.best_path().nodes.last().unwrap().pos, graph_node(2));
        assert!(matches!(
            session.advance(SearchSlice::unlimited()),
            SearchStatus::BudgetExhausted
        ));
    }

    #[test]
    fn zero_duration_slice_yields_without_consuming_work() {
        let grid = Grid::new();
        let context = MoveContext {
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let moves: Vec<Box<dyn Move>> = vec![Box::new(GraphMove::new(&[(0, 1, 1)], false))];
        let goal = BlockGoal::new(graph_node(1), 0);
        let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);

        assert!(matches!(
            session.advance(SearchSlice::new(100).with_time_budget(Duration::ZERO)),
            SearchStatus::InProgress
        ));
        assert_eq!(session.expansions(), 0);
        assert!(matches!(
            session.advance(SearchSlice::unlimited()),
            SearchStatus::Found(_)
        ));
    }

    #[test]
    fn inconsistent_admissible_heuristic_reopens_nodes_and_stays_optimal() {
        let grid = Grid::new();
        // A is expanded first and reaches C for 4. B has a high-but-admissible
        // heuristic, then improves C to 3 after C was already expanded.
        let moves: Vec<Box<dyn Move>> = vec![Box::new(GraphMove::new(
            &[(0, 1, 2), (0, 2, 2), (1, 3, 2), (2, 3, 1), (3, 4, 2)],
            true,
        ))];
        let goal = GraphGoal {
            target: 4,
            lower_bounds: [(0, 0), (1, 0), (2, 3), (3, 0), (4, 0)]
                .into_iter()
                .collect(),
        };
        let context = MoveContext {
            max_expansions: 100,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);
        let SearchStatus::Found(path) = session.advance(SearchSlice::unlimited()) else {
            panic!("graph search did not find the goal");
        };

        assert_eq!(path.total_cost, 5);
        assert_eq!(
            path.nodes.iter().map(|node| node.pos.x).collect::<Vec<_>>(),
            vec![0, 2, 3, 4]
        );
    }

    #[test]
    fn saturated_cost_is_reachable_and_does_not_confuse_stale_detection() {
        let grid = Grid::new();
        let moves: Vec<Box<dyn Move>> = vec![Box::new(GraphMove::new(
            &[(0, 1, Cost::MAX), (1, 2, 0)],
            false,
        ))];
        let goal = BlockGoal::new(graph_node(2), 0);
        let context = MoveContext {
            max_expansions: 10,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);
        let SearchStatus::Found(path) = session.advance(SearchSlice::unlimited()) else {
            panic!("saturated path was treated as unreachable");
        };

        assert_eq!(path.total_cost, Cost::MAX);
        assert_eq!(path.nodes.last().unwrap().pos, graph_node(2));
    }

    #[test]
    fn checked_revisions_reject_stale_work_without_mutating_the_session() {
        use std::sync::atomic::{AtomicU64, Ordering};

        struct MutableGoal {
            target: BlockPos,
            revision: AtomicU64,
        }

        impl Goal for MutableGoal {
            fn is_satisfied(&self, pos: BlockPos) -> bool {
                pos == self.target
            }

            fn revision(&self) -> u64 {
                self.revision.load(Ordering::Relaxed)
            }
        }

        let grid = Grid::new();
        let moves: Vec<Box<dyn Move>> = vec![Box::new(GraphMove::new(&[(0, 1, 1)], false))];
        let context = MoveContext {
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let goal = MutableGoal {
            target: graph_node(1),
            revision: AtomicU64::new(7),
        };
        let frozen = SearchAssumptions {
            world_revision: 11,
            cost_revision: 13,
        };
        let mut session = SearchSession::new_with_assumptions(
            &grid,
            graph_node(0),
            &goal,
            &moves,
            &context,
            frozen,
        );

        let status = session.advance_checked(
            SearchSlice::new(1),
            SearchAssumptions {
                world_revision: 12,
                ..frozen
            },
        );
        assert!(matches!(
            status,
            SearchStatus::Invalidated(RevisionMismatch {
                kind: RevisionKind::World,
                expected: 11,
                actual: 12,
            })
        ));
        assert_eq!(session.expansions(), 0);

        goal.revision.store(8, Ordering::Relaxed);
        assert!(matches!(
            session.advance(SearchSlice::new(0)),
            SearchStatus::InProgress
        ));
        assert_eq!(session.expansions(), 0);
        assert!(matches!(
            session.advance_checked(SearchSlice::new(1), frozen),
            SearchStatus::Invalidated(RevisionMismatch {
                kind: RevisionKind::Goal,
                expected: 7,
                actual: 8,
            })
        ));
        assert_eq!(session.expansions(), 0);
    }

    #[test]
    fn positive_advance_rejects_a_cached_terminal_after_goal_revision_changes() {
        use std::sync::atomic::{AtomicU64, Ordering};

        struct MutableGoal {
            target: BlockPos,
            revision: AtomicU64,
        }

        impl Goal for MutableGoal {
            fn is_satisfied(&self, pos: BlockPos) -> bool {
                pos == self.target
            }

            fn revision(&self) -> u64 {
                self.revision.load(Ordering::Relaxed)
            }
        }

        let grid = Grid::new();
        let context = MoveContext {
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let moves: Vec<Box<dyn Move>> = Vec::new();
        let goal = MutableGoal {
            target: graph_node(0),
            revision: AtomicU64::new(41),
        };
        let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);

        assert!(matches!(
            session.advance(SearchSlice::new(1)),
            SearchStatus::Found(_)
        ));
        goal.revision.store(42, Ordering::Relaxed);

        // A zero-work query remains observational and may report the cached
        // status, but any positive advance must check the revision first.
        assert!(matches!(
            session.advance(SearchSlice::new(0)),
            SearchStatus::Found(_)
        ));
        assert!(matches!(
            session.advance(SearchSlice::new(1)),
            SearchStatus::Invalidated(RevisionMismatch {
                kind: RevisionKind::Goal,
                expected: 41,
                actual: 42,
            })
        ));
    }

    #[test]
    fn cumulative_wall_clock_budget_fails_closed_before_accepting_a_late_goal() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct SlowGoalMove {
            called: Arc<AtomicBool>,
        }

        impl Move for SlowGoalMove {
            fn candidates(
                &self,
                from: BlockPos,
                _world: &dyn WorldView,
                _ctx: &MoveContext,
                out: &mut Vec<Edge>,
            ) {
                if from == graph_node(0) {
                    self.called.store(true, Ordering::Relaxed);
                    std::thread::sleep(Duration::from_millis(75));
                    out.push(Edge {
                        to: graph_node(1),
                        kind: MoveKind::Walk,
                        cost: 1,
                    });
                }
            }
        }

        let grid = Grid::new();
        let called = Arc::new(AtomicBool::new(false));
        let moves: Vec<Box<dyn Move>> = vec![Box::new(SlowGoalMove {
            called: Arc::clone(&called),
        })];
        let context = MoveContext {
            goal_tolerance: 0,
            max_expansions: 10,
            time_budget_ms: 50,
            ..ctx()
        };
        let goal = BlockGoal::new(graph_node(1), 0);
        let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);

        assert!(matches!(
            session.advance(SearchSlice::unlimited()),
            SearchStatus::BudgetExhausted
        ));
        assert!(called.load(Ordering::Relaxed));
        assert!(session.elapsed_compute() >= Duration::from_millis(50));
        assert_eq!(session.best_path().nodes.last().unwrap().pos, graph_node(0));
        assert!(matches!(
            session.advance(SearchSlice::unlimited()),
            SearchStatus::BudgetExhausted
        ));

        // An already-consumed cumulative budget is checked before the next
        // frontier pop, independently of the current slice's own allowance.
        let quick_moves: Vec<Box<dyn Move>> = vec![Box::new(GraphMove::new(&[(0, 1, 1)], false))];
        let mut already_spent =
            SearchSession::new(&grid, graph_node(0), &goal, &quick_moves, &context);
        already_spent.elapsed_compute = already_spent.hard_time_budget;
        assert!(matches!(
            already_spent.advance(SearchSlice::unlimited()),
            SearchStatus::BudgetExhausted
        ));
        assert_eq!(already_spent.expansions(), 0);
    }

    #[test]
    fn oversized_custom_move_is_rejected_without_truncation_or_successor_loss() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct OversizedMove;

        impl Move for OversizedMove {
            fn candidates(
                &self,
                from: BlockPos,
                _world: &dyn WorldView,
                _ctx: &MoveContext,
                out: &mut Vec<Edge>,
            ) {
                if from != graph_node(0) {
                    return;
                }
                for id in 0..=MAX_CANDIDATES_PER_MOVE {
                    out.push(Edge {
                        to: graph_node(id as i32 + 10),
                        kind: MoveKind::Walk,
                        cost: 1,
                    });
                }
            }
        }

        struct MustNotRun {
            called: Arc<AtomicBool>,
        }

        impl Move for MustNotRun {
            fn candidates(
                &self,
                from: BlockPos,
                _world: &dyn WorldView,
                _ctx: &MoveContext,
                out: &mut Vec<Edge>,
            ) {
                self.called.store(true, Ordering::Relaxed);
                if from == graph_node(0) {
                    out.push(Edge {
                        to: graph_node(1),
                        kind: MoveKind::Walk,
                        cost: 1,
                    });
                }
            }
        }

        let grid = Grid::new();
        let later_move_called = Arc::new(AtomicBool::new(false));
        let moves: Vec<Box<dyn Move>> = vec![
            Box::new(OversizedMove),
            Box::new(MustNotRun {
                called: Arc::clone(&later_move_called),
            }),
        ];
        let context = MoveContext {
            goal_tolerance: 0,
            max_expansions: 10,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let goal = BlockGoal::new(graph_node(1), 0);
        let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);

        assert!(matches!(
            session.advance(SearchSlice::unlimited()),
            SearchStatus::BudgetExhausted
        ));
        assert!(
            !later_move_called.load(Ordering::Relaxed),
            "the planner continued after rejecting oversized output"
        );
        assert_eq!(session.best_path().nodes.len(), 1);
        assert!(matches!(
            session.advance(SearchSlice::unlimited()),
            SearchStatus::BudgetExhausted
        ));
    }

    #[test]
    fn motion_plan_rejects_a_path_with_an_injected_wrong_parent() {
        let grid = Grid::new();
        let moves: Vec<Box<dyn Move>> = vec![Box::new(GraphMove::new(
            &[(0, 1, 1), (0, 2, 1), (2, 3, 1)],
            false,
        ))];
        let context = MoveContext {
            goal_tolerance: 0,
            max_expansions: 10,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let goal = BlockGoal::new(graph_node(3), 0);
        let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);
        let SearchStatus::Found(generated) = session.advance(SearchSlice::unlimited()) else {
            panic!("test graph did not produce a path");
        };
        assert!(session.motion_plan_for(generated).is_ok());

        // Node 3 was generated from node 2. Both node 1 and its metadata are
        // real, but splicing 1 in as node 3's parent must still be rejected.
        let forged = Path {
            nodes: vec![
                PathNode {
                    pos: graph_node(0),
                    reached_by: MoveKind::Start,
                },
                PathNode {
                    pos: graph_node(1),
                    reached_by: MoveKind::Walk,
                },
                PathNode {
                    pos: graph_node(3),
                    reached_by: MoveKind::Walk,
                },
            ],
            total_cost: 2,
        };
        assert_eq!(
            session.motion_plan_for(forged).unwrap_err(),
            MotionPlanError::MetadataMismatch
        );
    }

    #[test]
    fn adaptive_search_uses_component_costs_and_preserves_safety_tolls() {
        use crate::adaptive::{
            AdaptiveMode, AdaptiveProfile, AdaptiveRegime, AdaptiveSettings, AttributionEvidence,
            CostComponents, JourneyMetrics, MoveAttempt, MoveFeatures, ObservationContext,
            ObservationId, ObservationOutcome, PrimitiveId, ProfileKey, PromotionReport,
            TerrainClass, movement_costs_hash, next_observation_nonce,
        };
        use crate::local::moves::MovementCosts;

        let features = MoveFeatures::new(1, 0, TerrainClass::Unknown);
        let profile_key = ProfileKey::local_default();
        let observation_context = ObservationContext {
            profile: profile_key.clone(),
            journey_id: 71,
            generation: 1,
            leg: 0,
            plan_revision: 1,
            world_revision: 1,
            model_revision: 0,
            planner_settings_hash: 1,
            control_settings_hash: 1,
            baseline_costs_hash: movement_costs_hash(MovementCosts::default()),
            actor_capability_hash: profile_key.capability_hash,
            build_id: "astar-component-cost-test".into(),
            created_unix_ms: 1,
        };
        let regime = AdaptiveRegime {
            build_id: observation_context.build_id.clone(),
            planner_settings_hash: observation_context.planner_settings_hash,
            control_settings_hash: observation_context.control_settings_hash,
            baseline_costs_hash: observation_context.baseline_costs_hash,
            actor_capability_hash: observation_context.actor_capability_hash,
        };
        let settings = AdaptiveSettings {
            mode: AdaptiveMode::Enabled,
            min_samples: 3,
            max_buckets: 16,
            max_step_percent: 25,
            minimum_multiplier_ppm: 500_000,
            maximum_multiplier_ppm: 4_000_000,
            maximum_failure_upper_ppm: 900_000,
            minimum_evaluation_journeys: 1,
            maximum_p95_regression_percent: 10,
        };
        let mut profile = AdaptiveProfile::new(profile_key);
        for edge_index in 0_u32..3 {
            let attempt = MoveAttempt::begin(
                observation_context.clone(),
                ObservationId {
                    nonce: next_observation_nonce(),
                    journey_id: observation_context.journey_id,
                    generation: observation_context.generation,
                    leg: observation_context.leg,
                    edge_index,
                    attempt: 0,
                },
                PrimitiveId::WALK_CARDINAL,
                features,
                graph_node(0),
                graph_node(3),
                CostComponents {
                    time: 10,
                    ..CostComponents::default()
                },
                10,
                0,
            )
            .unwrap();
            let observation = attempt
                .finish(
                    40,
                    u64::from(edge_index) + 1,
                    ObservationOutcome::Success,
                    AttributionEvidence::ReachedPlannedNode,
                )
                .unwrap();
            assert!(profile.observe(&observation, &settings));
        }
        let metrics = JourneyMetrics {
            journeys: 1,
            failed: 0,
            p95_ticks: 100,
            damage_half_hearts: 0,
            setbacks: 0,
            corrections: 0,
            disconnects: 0,
        };
        let report = PromotionReport {
            evaluation_id: 1,
            candidate_data_revision: profile.data_revision,
            expected_model_revision: profile.model_revision(),
            candidate_hash: profile.candidate_hash(&settings),
            known_good: metrics,
            shadow: metrics,
        };
        profile.promote(&report, &settings).unwrap();
        let model = profile.snapshot(MovementCosts::default(), &settings, &regime);

        let direct_components = CostComponents {
            time: 10,
            safety: 100,
            ..CostComponents::default()
        };
        let adapted =
            model.edge_components(PrimitiveId::WALK_CARDINAL, features, direct_components);
        assert_eq!(adapted.time, 13);
        assert_eq!(adapted.safety, 100, "immutable safety toll was scaled");
        assert_eq!(adapted.total(), 113);

        struct ComponentGraphMove {
            walk_features: MoveFeatures,
        }

        impl Move for ComponentGraphMove {
            fn candidates(
                &self,
                from: BlockPos,
                _world: &dyn WorldView,
                _ctx: &MoveContext,
                out: &mut Vec<Edge>,
            ) {
                match from.x {
                    0 => {
                        out.push(Edge {
                            to: graph_node(3),
                            kind: MoveKind::Walk,
                            cost: 110,
                        });
                        out.push(Edge {
                            to: graph_node(1),
                            kind: MoveKind::Jump,
                            cost: 50,
                        });
                    }
                    1 => out.push(Edge {
                        to: graph_node(3),
                        kind: MoveKind::Jump,
                        cost: 62,
                    }),
                    _ => {}
                }
            }

            fn metadata(
                &self,
                from: BlockPos,
                edge: &Edge,
                _world: &dyn WorldView,
                _ctx: &MoveContext,
            ) -> Option<MotionMetadata> {
                let (primitive, features, predicted) =
                    if from == graph_node(0) && edge.to == graph_node(3) {
                        (
                            PrimitiveId::WALK_CARDINAL,
                            self.walk_features,
                            CostComponents {
                                time: 10,
                                safety: 100,
                                ..CostComponents::default()
                            },
                        )
                    } else {
                        (
                            PrimitiveId::JUMP,
                            MoveFeatures::new(1, 0, TerrainClass::Unknown),
                            CostComponents {
                                time: edge.cost,
                                ..CostComponents::default()
                            },
                        )
                    };
                Some(MotionMetadata {
                    primitive,
                    features,
                    predicted,
                    predicted_ticks: 10,
                })
            }
        }

        let grid = Grid::new();
        let moves: Vec<Box<dyn Move>> = vec![Box::new(ComponentGraphMove {
            walk_features: features,
        })];
        let context = MoveContext {
            goal_tolerance: 0,
            max_expansions: 10,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let goal = BlockGoal::new(graph_node(3), 0);

        let mut baseline = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);
        let SearchStatus::Found(baseline_path) = baseline.advance(SearchSlice::unlimited()) else {
            panic!("baseline graph did not produce a path");
        };
        assert_eq!(baseline_path.total_cost, 110);
        assert_eq!(
            baseline_path
                .nodes
                .iter()
                .map(|node| node.pos.x)
                .collect::<Vec<_>>(),
            vec![0, 3]
        );

        let mut adaptive = SearchSession::new_with_cost_model(
            &grid,
            graph_node(0),
            &goal,
            &moves,
            &context,
            &model,
        );
        let SearchStatus::Found(adaptive_path) = adaptive.advance(SearchSlice::unlimited()) else {
            panic!("adaptive graph did not produce a path");
        };
        assert_eq!(adaptive_path.total_cost, 112);
        assert_eq!(
            adaptive_path
                .nodes
                .iter()
                .map(|node| node.pos.x)
                .collect::<Vec<_>>(),
            vec![0, 1, 3],
            "component-aware adaptive cost was not used for route selection"
        );
    }

    fn reference_dijkstra(
        node_count: i32,
        edges: &[(i32, i32, Cost)],
        start: i32,
        goal: i32,
    ) -> Option<Cost> {
        let mut outgoing: Vec<Vec<(i32, Cost)>> = vec![Vec::new(); node_count as usize];
        for &(from, to, cost) in edges {
            outgoing[from as usize].push((to, cost));
        }
        let mut distances: Vec<Option<Cost>> = vec![None; node_count as usize];
        let mut frontier = BinaryHeap::new();
        distances[start as usize] = Some(0);
        frontier.push(Reverse((0, start)));
        while let Some(Reverse((cost, at))) = frontier.pop() {
            if distances[at as usize] != Some(cost) {
                continue;
            }
            if at == goal {
                return Some(cost);
            }
            for &(to, edge_cost) in &outgoing[at as usize] {
                let next = cost.saturating_add(edge_cost);
                if distances[to as usize].is_none_or(|known| next < known) {
                    distances[to as usize] = Some(next);
                    frontier.push(Reverse((next, to)));
                }
            }
        }
        None
    }

    #[test]
    fn randomized_graph_search_matches_independent_dijkstra() {
        let grid = Grid::new();
        let context = MoveContext {
            goal_tolerance: 0,
            max_expansions: 10_000,
            time_budget_ms: u64::MAX,
            ..ctx()
        };
        let mut random = 0x9E37_79B9_7F4A_7C15u64;

        for case in 0..96 {
            let node_count = 4 + (case % 9);
            let mut edges = Vec::new();
            for from in 0..node_count {
                for to in 0..node_count {
                    random = random
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    if from != to && random >> 61 != 0 {
                        let cost = 1 + ((random >> 32) as Cost % 50);
                        edges.push((from, to, cost));
                    }
                }
            }
            let expected = reference_dijkstra(node_count, &edges, 0, node_count - 1);
            let moves: Vec<Box<dyn Move>> = vec![Box::new(GraphMove::new(&edges, false))];
            let goal = BlockGoal::new(graph_node(node_count - 1), 0);
            let mut session = SearchSession::new(&grid, graph_node(0), &goal, &moves, &context);

            match (session.advance(SearchSlice::unlimited()), expected) {
                (SearchStatus::Found(path), Some(cost)) => {
                    assert_eq!(path.total_cost, cost, "random graph case {case}");
                }
                (SearchStatus::Exhausted, None) => {}
                (actual, expected) => {
                    panic!("random graph case {case}: got {actual:?}, expected {expected:?}")
                }
            }
        }
    }
}
