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

/// Controls whether the planner may route the bot through water.
///
/// This exists for the same reason [`LavaPolicy`] does, but the hazard is the
/// server rather than the world. Azalea's in-water physics disagrees with the
/// vanilla server every single tick: a bot that merely sits in a pool, with no
/// navigation running at all, accumulates a Grim `Simulation` offset on every
/// tick and is kicked within seconds. Land movement over the same distance
/// produces a handful of flags in total. Until the physics is fixed, a planned
/// swim is a planned disconnect, so the default is to refuse one.
///
/// Escaping water is always permitted, exactly as a bot that starts inside the
/// lava buffer may still walk out of it. Refusing that would strand any bot
/// that spawned or fell in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WaterPolicy {
    /// Never plan a move that ends with the feet in water. From inside water,
    /// only moves that leave it are offered.
    #[default]
    Forbidden,
    /// Allow swimming, with [`MoveContext::water_penalty`] added per edge.
    Penalized,
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
    /// Cost of a running jump across a gap, per block of horizontal distance.
    ///
    /// Deliberately dearer than walking the same distance: if there is a way
    /// round, walking round is the safer path, and a mistimed gap jump costs
    /// far more than the few ticks it saves.
    pub parkour_per_block: Cost,
    /// Cost of one block of swimming, horizontal or vertical.
    ///
    /// Swimming is roughly half of walking speed in vanilla, and it must never
    /// be set below [`Self::cardinal_walk`]: the A* heuristic assumes a walk is
    /// the cheapest way to cross a block, so a cheap swim would make it
    /// inadmissible.
    pub swim: Cost,
    /// Cost of one block of ladder, up or down.
    ///
    /// Dear compared with walking because climbing is slow (about a fifth of
    /// walking speed going up), so a ladder is worth taking only when there is
    /// no way round, which on a sheer face there is not.
    pub climb: Cost,
    /// Cost of climbing out of water onto a bank one block above the surface.
    ///
    /// Dearer than a jump because swimming up is a slow drift rather than a
    /// launch, and a bot that mistimes it falls back into the water and has to
    /// start the climb again.
    pub swim_exit: Cost,
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
            parkour_per_block: 22,
            swim: 20,
            climb: 30,
            swim_exit: 45,
        }
    }
}

/// Tuning knobs for a single path search.
#[derive(Debug, Clone)]
pub struct MoveContext {
    /// Maximum blocks the bot may drop in one fall move.
    ///
    /// Three was "the tallest drop that hurts nobody", which is the wrong
    /// question. A player coming down a mountain jumps off ledges constantly
    /// and accepts the scratch; the hub's terraces step down five to ten blocks
    /// at a time, so at three the bot could climb the mountain and then had no
    /// legal move back off it and stood on top. Falls are priced by the damage
    /// they do (see [`Self::fall_damage_penalty`]) and clamped to what the bot
    /// can survive, so "how far may I drop" is a cost question, not a ban.
    ///
    /// Candidate generation sanitizes this to `0..=127`: non-positive values
    /// disable drops, while larger values use 127. The upper bound is the
    /// largest downward distance shared by the signed parkour encoding and a
    /// bounded amount of work per movement rule.
    pub max_fall: i32,
    /// Extra cost per half-heart of fall damage a drop would cause.
    ///
    /// What stops the bot treating every cliff as a shortcut. A drop hurts for
    /// each block past [`SAFE_FALL`], so this converts "that will cost me three
    /// hearts" into blocks-of-walking the search can weigh against going round.
    pub fall_damage_penalty: Cost,
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
    /// Whether water is forbidden or merely costly.
    pub water_policy: WaterPolicy,
    /// Toll for standing on a partial block with a taller partial block beside
    /// it. See `grazing_step_penalty` in the search.
    ///
    /// Five blocks of walking: enough to prefer the even part of a snowfield,
    /// far too little to refuse a mountain made of the stuff. Set to 0 to
    /// restore the previous behaviour exactly.
    pub grazing_step_penalty: Cost,
    /// Extra cost per swim edge, used only by [`WaterPolicy::Penalized`].
    ///
    /// Large on purpose. Under `Penalized` the intent is "swim only when there
    /// is genuinely no way round", not "swim when it is a bit shorter".
    pub water_penalty: Cost,

    /// Seed used to vary ties between equal-cost paths.
    pub path_seed: u64,

    /// Extra cost charged for standing on specific blocks.
    ///
    /// This is how the navigator remembers a route that did not work. A search
    /// is deterministic, so a leg that ends Stuck replans from the same block
    /// and gets the same path back; re-seeding the tie-break only reorders
    /// equal-cost options and a route that is genuinely shortest stays
    /// shortest. Six legs in a row once burned three minutes on one doorway
    /// this way. Charging the block that failed makes the next search prefer
    /// the way round, which is what a person does after walking into a wall.
    ///
    /// A toll rather than a wall, deliberately: if the failed block is the only
    /// way through, the bot should still try it rather than declare the goal
    /// unreachable.
    pub avoid: std::sync::Arc<std::collections::HashMap<BlockPos, Cost>>,
}

impl Default for MoveContext {
    fn default() -> Self {
        Self {
            max_fall: 10,
            goal_tolerance: 1,
            max_expansions: 150_000,
            time_budget_ms: 2_000,
            costs: MovementCosts::default(),
            wall_penalty: 2,
            lava_policy: LavaPolicy::default(),
            lava_penalty: 10_000,
            lava_proximity_radius: 4,
            lava_proximity_penalty: 6,
            water_policy: WaterPolicy::default(),
            water_penalty: 400,
            grazing_step_penalty: 50,
            fall_damage_penalty: 45,
            path_seed: 0,
            avoid: std::sync::Arc::default(),
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
    ///
    /// Implementations are trusted cooperative extensions: one call must
    /// terminate promptly and return no more than
    /// [`crate::local::astar::MAX_CANDIDATES_PER_MOVE`] candidates. The search
    /// rejects oversized output, but cannot preempt code while this method is
    /// running.
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

    /// Generator-authored metadata for adaptive costing and execution
    /// telemetry. Custom moves may return `None`; they remain fully supported
    /// but are neither learned nor assigned a built-in heuristic.
    fn metadata(
        &self,
        _from: BlockPos,
        _edge: &Edge,
        _world: &dyn WorldView,
        _ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        None
    }
}

/// Standard walking, jumping, falling, and teleport extension points.
pub fn default_moves() -> Vec<Box<dyn Move>> {
    vec![
        Box::new(WalkMove),
        Box::new(JumpMove),
        Box::new(ClimbMove),
        Box::new(ParkourMove),
        Box::new(SwimMove),
        Box::new(FallMove),
        Box::new(super::teleport::AotvMove),
        Box::new(super::teleport::EtherwarpMove),
    ]
}

const CARDINALS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAGONALS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
pub const MAX_LAVA_SCAN_RADIUS: i32 = 32;

/// Whether the bot's feet are in water at `pos`, and so swimming rather than
/// standing on anything.
fn in_water(world: &dyn WorldView, pos: BlockPos) -> bool {
    world.block(pos) == BlockKind::Water
}

/// Whether the bot's feet are on a ladder, vine or scaffolding at `pos`.
fn on_climbable(world: &dyn WorldView, pos: BlockPos) -> bool {
    world.block(pos) == BlockKind::Climbable
}

/// Whether `pos` is empty space as far as the body is concerned.
///
/// Every "is this air" test in this file means this. A block with no collision
/// box is walked through exactly like air, so testing `== Air` instead of this
/// treats a tuft of grass as a wall.
fn open(world: &dyn WorldView, pos: BlockPos) -> bool {
    matches!(world.block(pos), BlockKind::Air | BlockKind::Passable)
}

/// Whether `pos` is somewhere the bot stands on its feet rather than floats or
/// hangs.
///
/// Walking, jumping and parkour all need real ground contact, so they ask this
/// instead of [`WorldView::standable`], which also accepts open water and
/// ladders. Those two are owned by [`SwimMove`] and [`ClimbMove`], which know
/// what the body is actually doing there.
fn dry_standable(world: &dyn WorldView, pos: BlockPos) -> bool {
    !in_water(world, pos) && !on_climbable(world, pos) && world.standable(pos)
}

/// Standable neighbours counted as "open" enough for the wall toll to reach
/// full strength. A block in the open has all eight; a one-wide corridor has
/// about two. Below this the toll fades, above it is capped.
const OPEN_ROOM: Cost = 6;

/// Adds a cost for hugging a wall, scaled by how much room there is to not.
///
/// A flat per-wall toll is wrong in a corridor: every block there is against a
/// wall, so the toll is the same everywhere and cannot pull the bot off the
/// wall - there is nowhere to go. All it does is inflate the route's cost and,
/// worse, tempt a detour that trades real distance for imaginary clearance.
///
/// So the toll is scaled by the standable room around the feet. In the open,
/// where stepping one block over really does move the bot off the wall, it is
/// full strength and pulls it to the centre line. In a one-wide passage, where
/// there is no centre to find, it fades to almost nothing and the bot just
/// walks the passage. This is the "depends on how many blocks he can use" rule:
/// the penalty for being near a wall is only as large as the freedom to avoid
/// it.
pub fn wall_proximity_penalty(pos: BlockPos, world: &dyn WorldView, per_wall: Cost) -> Cost {
    if per_wall == 0 {
        return 0;
    }
    let is_wall = |b: BlockKind| matches!(b, BlockKind::Solid | BlockKind::Fence);
    let mut walls: Cost = 0;
    for (dx, dz) in CARDINALS {
        if is_wall(world.block(offset(pos, dx, 0, dz)))
            || is_wall(world.block(offset(pos, dx, 1, dz)))
        {
            walls += 1;
        }
    }
    // No walls: already central, and this skips the room scan for the vast
    // majority of nodes, which are nowhere near anything.
    if walls == 0 {
        return 0;
    }
    let mut room: Cost = 0;
    for (dx, dz) in CARDINALS.into_iter().chain(DIAGONALS) {
        if world.standable(offset(pos, dx, 0, dz)) {
            room += 1;
        }
    }
    // walls * per_wall, scaled by (room capped at OPEN_ROOM) / OPEN_ROOM.
    // Widen before multiplying so division is applied to the mathematical
    // product rather than to an already-saturated u32 intermediate.
    let scaled = u64::from(per_wall)
        .saturating_mul(u64::from(walls))
        .saturating_mul(u64::from(room.min(OPEN_ROOM)))
        / u64::from(OPEN_ROOM);
    scaled.min(u64::from(Cost::MAX)) as Cost
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
    let clearance = clearance.clamp(0, MAX_LAVA_SCAN_RADIUS);
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
    let radius = radius.min(MAX_LAVA_SCAN_RADIUS);

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
        // Feet in water means swimming, not walking, and feet on a ladder mean
        // climbing. Both belong to the rule that knows what the body is doing,
        // and both are charged at their own slower rate.
        if in_water(world, from) || on_climbable(world, from) {
            return;
        }
        for (dx, dz) in CARDINALS {
            let to = offset(from, dx, 0, dz);
            if dry_standable(world, to) {
                out.push(Edge {
                    to,
                    kind: MoveKind::Walk,
                    cost: ctx.costs.cardinal_walk,
                });
            }
        }
        let body_clear = |p: BlockPos| open(world, p) && open(world, offset(p, 0, 1, 0));
        for (dx, dz) in DIAGONALS {
            let to = offset(from, dx, 0, dz);
            if dry_standable(world, to)
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
        let from_step = matches!(world.block(from), BlockKind::Step(_));
        for (dx, dz) in CARDINALS {
            for dy in [1, -1] {
                let to = offset(from, dx, dy, dz);
                if !dry_standable(world, to) {
                    continue;
                }
                let to_step = matches!(world.block(to), BlockKind::Step(_));
                let walkable = match dy {
                    1 => from_step,
                    _ => from_step || to_step,
                };
                if !walkable {
                    continue;
                }
                if dy == 1 && !open(world, offset(from, 0, 2, 0)) {
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

    fn metadata(
        &self,
        from: BlockPos,
        edge: &Edge,
        world: &dyn WorldView,
        ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        let primitive = if from.y != edge.to.y {
            crate::PrimitiveId::STEP
        } else if from.x != edge.to.x && from.z != edge.to.z {
            crate::PrimitiveId::WALK_DIAGONAL
        } else {
            crate::PrimitiveId::WALK_CARDINAL
        };
        Some(crate::planning::metadata_from_generated_edge(
            primitive, from, edge.to, edge.kind, world, ctx, edge.cost,
        ))
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
        // In water the jump key is a slow upward drift, not a launch off the
        // ground, so a block-high hop out of a pool is not a jump: SwimMove
        // plans that climb with the cost it really takes. On a ladder the same
        // key is the climb input, and ClimbMove owns that.
        if in_water(world, from) || on_climbable(world, from) {
            return;
        }
        if !open(world, offset(from, 0, 2, 0)) {
            return;
        }
        for (dx, dz) in CARDINALS {
            let to = offset(from, dx, 1, dz);

            if dry_standable(world, to) {
                // A slab or stair only raises the body half a block, so getting
                // onto one is closer to a step than to a full hop even when its
                // block sits a level up. Charging it the cheaper `step` makes
                // the planner route up a flight of slabs rather than pillar-hop
                // full blocks beside it - which is what the slabs are for. The
                // move is still a Jump so the follower does hop; only the price
                // the search pays for it changes.
                let onto_step = matches!(world.block(to), BlockKind::Step(_));
                out.push(Edge {
                    to,
                    kind: MoveKind::Jump,
                    cost: if onto_step {
                        ctx.costs.step
                    } else {
                        ctx.costs.jump
                    },
                });
            }
        }
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }

    fn metadata(
        &self,
        from: BlockPos,
        edge: &Edge,
        world: &dyn WorldView,
        ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        Some(crate::planning::metadata_from_generated_edge(
            crate::PrimitiveId::JUMP,
            from,
            edge.to,
            edge.kind,
            world,
            ctx,
            edge.cost,
        ))
    }
}

/// Climb a ladder, vine or scaffolding, and step on and off it.
///
/// This is the only rule that gains height without a jump, and on a hand-built
/// map it is routinely the only way up at all. The measured case: the summit at
/// (281,163,336) is reachable from the ground by no combination of walking,
/// jumping, parkour and falling, because every approach ends in a five to
/// seven block sheer riser. Two ladders bridge it, at (289,143..147,327) and
/// (277,150..155,341), and with them the summit becomes reachable and without
/// them it does not. A search cannot find a move it does not have.
pub struct ClimbMove;

impl Move for ClimbMove {
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        ctx: &MoveContext,
        out: &mut Vec<Edge>,
    ) {
        let mut push = |to: BlockPos| {
            out.push(Edge {
                to,
                kind: MoveKind::Climb,
                cost: ctx.costs.climb,
            })
        };

        if !on_climbable(world, from) {
            // Getting on. A ladder is grabbed from the side, level with the
            // feet or one block up, which is the rung a standing player can
            // reach. Swimming onto a ladder is left out: the two physics
            // states fight, and water is forbidden by default anyway.
            if in_water(world, from) {
                return;
            }
            for (dx, dz) in CARDINALS {
                for dy in [0, 1] {
                    let to = offset(from, dx, dy, dz);
                    if on_climbable(world, to) && world.standable(to) {
                        // Stepping straight into the ladder and reaching up to
                        // the rung above it are not the same difficulty, and
                        // costing them the same made the planner pick the hard
                        // one whenever both existed - it saves a node. Walking
                        // in is one held key; grabbing the higher rung means
                        // rising and moving sideways on the same tick, into a
                        // block whose collision box is a sliver against the
                        // wall you are trying to get onto. The bot wedged
                        // itself on that sliver at the foot of the ladder and
                        // gave up with a complete route in hand.
                        //
                        // Both stay available, because a ladder that starts
                        // above the floor can only be entered the hard way.
                        //
                        // The penalty has to be worth more than a rung. The
                        // high grab starts a rung further up, so it saves a
                        // whole climb step, and any penalty smaller than that
                        // still comes out cheaper and gets chosen anyway -
                        // which is what happened with a penalty of one jump.
                        out.push(Edge {
                            to,
                            kind: MoveKind::Climb,
                            cost: ctx.costs.climb.saturating_add(if dy == 1 {
                                ctx.costs.climb.saturating_add(ctx.costs.jump)
                            } else {
                                0
                            }),
                        });
                    }
                }
            }
            return;
        }

        // On the ladder: one rung at a time, up or down. Vanilla climbs by
        // holding the jump key and descends by letting go, so both directions
        // are a single sustained input and neither needs timing.
        for dy in [1, -1] {
            let to = offset(from, 0, dy, 0);
            if on_climbable(world, to) && world.standable(to) {
                push(to);
            }
        }

        // Getting off, onto a ledge level with the feet or one rung up. The
        // rung above has to be clear to step out of the top of a ladder, which
        // is the normal case: the topmost rung is the one level with the floor
        // it serves.
        for (dx, dz) in CARDINALS {
            for dy in [0, 1] {
                let to = offset(from, dx, dy, dz);
                if !dry_standable(world, to) {
                    continue;
                }
                if dy == 1 && !open(world, offset(from, 0, 2, 0)) {
                    continue; // no headroom to rise
                }
                push(to);
            }
        }
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }

    fn metadata(
        &self,
        from: BlockPos,
        edge: &Edge,
        world: &dyn WorldView,
        ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        Some(crate::planning::metadata_from_generated_edge(
            crate::PrimitiveId::CLIMB,
            from,
            edge.to,
            edge.kind,
            world,
            ctx,
            edge.cost,
        ))
    }
}

/// Jump a gap, landing 2 or 3 blocks away in a cardinal direction.
///
/// A vanilla sprint jump clears about 4 blocks of travel, so landing 3 blocks
/// out (a 2 block gap) is comfortably inside what the server will simulate.
/// The 4 block version exists in vanilla but only from a full run-up, and a
/// pathfinder cannot promise the run-up, so it is left out.
///
/// Every intermediate column must be *empty at foot level*, otherwise this is
/// not a gap and [`WalkMove`] or [`JumpMove`] already covers it more cheaply.
pub struct ParkourMove;

impl Move for ParkourMove {
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        ctx: &MoveContext,
        out: &mut Vec<Edge>,
    ) {
        // No gap jump starts in water or off a ladder. A sprint jump is all
        // takeoff velocity, and neither a swimmer nor a climber has any:
        // vanilla gives an upward drift instead, so a planned arc out of a pool
        // or off a rung simply does not happen.
        if in_water(world, from) || on_climbable(world, from) {
            return;
        }
        // Jumping needs two blocks of headroom over the takeoff.
        if !passable(world, offset(from, 0, 1, 0)) || !passable(world, offset(from, 0, 2, 0)) {
            return;
        }
        // Jumping off a slab or stair is fine, and stepping stones in the air
        // often have nothing underneath them at all, so requiring solid ground
        // below the takeoff silently disables parkour across exactly the
        // terrain that needs it.
        let debug = crate::debug::parkour_enabled();
        if debug {
            eprintln!(
                "parkour@{:?}: here={:?} below={:?} up1={:?} up2={:?} east={:?} east2={:?} east3={:?} east4={:?}",
                from,
                world.block(from),
                world.block(offset(from, 0, -1, 0)),
                world.block(offset(from, 0, 1, 0)),
                world.block(offset(from, 0, 2, 0)),
                world.block(offset(from, 1, 0, 0)),
                world.block(offset(from, 2, 0, 0)),
                world.block(offset(from, 3, 0, 0)),
                world.block(offset(from, 4, 0, 0)),
            );
        }
        let on_step = matches!(world.block(from), BlockKind::Step(_));
        if !on_step && world.block(offset(from, 0, -1, 0)) != BlockKind::Solid {
            if debug {
                eprintln!("  bail: no solid ground under takeoff");
            }
            return;
        }

        for (ox, oz) in jump_offsets() {
            let distance = ((ox * ox + oz * oz) as f64).sqrt();
            // Landing level or one block up. Real terrain almost never offers
            // a flat gap: the far side of a broken bridge is usually a step
            // higher, and refusing that is why a bot stands at the edge doing
            // nothing. Dropping is left to FallMove, which costs falls
            // properly.
            for rise in [0, 1] {
                // Measured, not assumed: a full-length jump with a rise looks
                // plannable but the bot lands in the gap every time, because
                // the whole arc is spent going forward with nothing left to
                // climb with. Rising jumps are therefore shorter.
                //
                // A run-up buys distance, so it sets the limit rather than
                // deciding whether to jump at all. It used to be a veto - no
                // block behind the takeoff meant no jump past a short hop and
                // no rising jump ever - on the reasoning that a player could
                // not make those either. That is wrong, and a hand-built
                // parkour course proves it: every platform on one is a single
                // block with nothing behind it, and the jumps between them are
                // ordinary two-block gaps that a standing sprint jump clears
                // comfortably. The veto refused every jump on the course, so
                // the whole thing was invisible to the planner and a bot sent
                // to the far end walked underneath it and reported that there
                // was no route.
                let run_up = has_run_up(world, from, ox, oz, 1);
                let limit = match (rise > 0, run_up) {
                    (false, true) => MAX_FLAT_JUMP,
                    (true, true) => MAX_RISING_JUMP,
                    (false, false) => MAX_STANDING_JUMP,
                    (true, false) => MAX_STANDING_RISING_JUMP,
                };
                if distance > limit {
                    if debug {
                        eprintln!(
                            "  skip {ox},{oz}/{rise}: {distance:.2} over {limit:.2} \
                             (run_up={run_up})"
                        );
                    }
                    continue;
                }
                let to = offset(from, ox, rise, oz);
                if !dry_standable(world, to) {
                    continue;
                }
                // The whole line has to be a real gap: nothing to stand on
                // (or WalkMove already covers it more cheaply) and nothing in
                // the way of an arc that peaks above the higher end.
                let head = 2 + rise;
                let clear = crossed_columns(from, ox, oz).all(|mid| {
                    !world.standable(mid)
                        && (0..=head).all(|dy| passable(world, offset(mid, 0, dy, 0)))
                });
                if !clear {
                    continue;
                }
                // Room to land without clipping our head, and room to get our
                // head over the lip on the way in.
                if !passable(world, offset(to, 0, 1, 0)) {
                    continue;
                }
                if rise > 0 && !passable(world, offset(from, 0, head, 0)) {
                    continue;
                }
                // The longest jumps need more than one block of run-up, not
                // just any. This still bites only above 3.5, which is further
                // than a standing jump can reach anyway.
                let blocks = distance.round() as i32;
                if distance > 3.5 && !has_run_up(world, from, ox, oz, 2) {
                    if debug {
                        eprintln!("  skip {ox},{oz}/{rise}: run-up too short");
                    }
                    continue;
                }
                if debug {
                    eprintln!("  edge: {ox},{oz} dist {distance:.2} rise {rise} -> {to:?}");
                }
                out.push(Edge {
                    to,
                    kind: MoveKind::Parkour {
                        blocks: blocks as u8,
                        rise: rise as i8,
                    },
                    cost: ctx
                        .costs
                        .parkour_per_block
                        .saturating_mul(blocks as Cost)
                        .saturating_add(ctx.costs.jump.saturating_mul(rise as Cost)),
                });
            }

            // Gap jumps that land *lower*. A player coming down a mountain
            // jumps a ravine and lets the fall carry them across; the far side
            // being lower makes the jump easier, not harder.
            //
            // "Dropping is left to FallMove" was wrong, because FallMove steps
            // one block sideways and then drops straight: it cannot cross a gap
            // at all. So a terrace climbed by a rising parkour jump had no
            // reverse move, and a bot that walked up the mountain had no legal
            // way back off it and stood on the summit.
            if distance > MAX_FLAT_JUMP {
                continue;
            }
            for drop in 1..=generated_drop_limit(ctx.max_fall) {
                let to = offset(from, ox, -drop, oz);
                if !dry_standable(world, to) {
                    // Keep descending through clear air until the first viable
                    // landing. Water, a ladder, unloaded space, or collision
                    // at either body block ends the arc.
                    if !open(world, to) || !open(world, offset(to, 0, 1, 0)) {
                        break;
                    }
                    continue;
                }
                // The crossed columns must be a real gap for the whole descent,
                // not just at takeoff height: if any of them can be stood on on
                // the way past, walking down is cheaper and safer than jumping.
                let clear = crossed_columns(from, ox, oz).all(|mid| {
                    (-drop..=2).all(|dy| passable(world, offset(mid, 0, dy, 0)))
                        && (-drop..=0).all(|dy| !world.standable(offset(mid, 0, dy, 0)))
                });
                if !clear {
                    break;
                }
                if !passable(world, offset(to, 0, 1, 0)) {
                    break;
                }
                let blocks = distance.round() as i32;
                if blocks >= 3 && !has_run_up(world, from, ox, oz, 1) {
                    break;
                }
                let hurt = if in_water(world, to) {
                    0
                } else {
                    (drop - SAFE_FALL).max(0) as Cost
                };
                let Ok(encoded_rise) = i8::try_from(-drop) else {
                    // `generated_drop_limit` makes this unreachable, but keep
                    // malformed future bounds from silently wrapping an edge.
                    break;
                };
                out.push(Edge {
                    to,
                    kind: MoveKind::Parkour {
                        blocks: blocks as u8,
                        rise: encoded_rise,
                    },
                    cost: ctx
                        .costs
                        .parkour_per_block
                        .saturating_mul(blocks as Cost)
                        .saturating_add(ctx.costs.fall_per_block.saturating_mul(drop as Cost))
                        .saturating_add(ctx.fall_damage_penalty.saturating_mul(hurt)),
                });
                break;
            }
        }
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }

    fn metadata(
        &self,
        from: BlockPos,
        edge: &Edge,
        world: &dyn WorldView,
        ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        Some(crate::planning::metadata_from_generated_edge(
            crate::PrimitiveId::PARKOUR,
            from,
            edge.to,
            edge.kind,
            world,
            ctx,
            edge.cost,
        ))
    }
}

/// Longest flat jump, in blocks of travel. A vanilla sprint jump clears a 3
/// block gap, which is 4 blocks from takeoff centre to landing centre.
const MAX_FLAT_JUMP: f64 = 4.0;
/// Longest jump that also gains a block of height. Climbing eats into the arc,
/// so this is shorter than [`MAX_FLAT_JUMP`]; the value is what the bot
/// actually lands, not what looks plausible on paper.
const MAX_RISING_JUMP: f64 = 3.2;
/// Longest jump from a standstill, with no room to build speed first.
///
/// A sprint jump's forward impulse is applied at takeoff and does not need
/// momentum already in hand, so an isolated block is not a dead end - it just
/// costs the acceleration a run-up would have added. Three blocks of travel is
/// the ordinary two-block gap that parkour is built from; the four-block jump
/// really does need a runway, and is still refused without one.
const MAX_STANDING_JUMP: f64 = 3.05;
/// The same, while also gaining a block. Climbing eats into the arc here just
/// as it does with a run-up, but a two-block gap one block up is a standard
/// jump rather than an exotic one.
const MAX_STANDING_RISING_JUMP: f64 = 3.05;

/// Every landing offset worth considering, nearest first.
///
/// Cardinal-only jumps miss most real terrain: stepping stones, ruined bridges
/// and rock faces put the next foothold off-axis, and a bot that will only
/// jump along x or z simply stops at the edge. Offsets shorter than two blocks
/// are left out because walking or stepping up already covers them, and more
/// cheaply.
fn jump_offsets() -> impl Iterator<Item = (i32, i32)> {
    let reach = MAX_FLAT_JUMP.ceil() as i32;
    (-reach..=reach).flat_map(move |ox| {
        (-reach..=reach).filter_map(move |oz| {
            let distance = ((ox * ox + oz * oz) as f64).sqrt();
            (2.0..=MAX_FLAT_JUMP)
                .contains(&distance)
                .then_some((ox, oz))
        })
    })
}

/// The block columns a jump passes over, excluding both ends.
///
/// Sampled along the straight line rather than stepped per axis, so a diagonal
/// jump checks the columns it actually flies over instead of an L-shaped path
/// it never takes.
fn crossed_columns(from: BlockPos, ox: i32, oz: i32) -> impl Iterator<Item = BlockPos> {
    let distance = ((ox * ox + oz * oz) as f64).sqrt();
    let steps = (distance * 4.0).ceil() as i32;
    let landing = offset(from, ox, 0, oz);
    let mut seen = Vec::new();
    for step in 1..steps {
        let t = step as f64 / steps as f64;
        let x = (ox as f64 * t).round() as i32;
        let z = (oz as f64 * t).round() as i32;
        let pos = offset(from, x, 0, z);
        // Both ends are checked separately: the takeoff is where we already
        // stand, and the landing is standable on purpose, so including it here
        // rejects every jump.
        if pos != from && pos != landing && !seen.contains(&pos) {
            seen.push(pos);
        }
    }
    seen.into_iter()
}

/// Whether there is somewhere to run up from, `back` blocks behind the takeoff.
///
/// The approach is rarely level with the takeoff: slabs, stairs and stepping
/// stones all put it half or a whole block off, and a bot arriving over those
/// is running just as fast. Requiring an exactly level run-up rejects nearly
/// every real gap, so accept one block either side.
fn has_run_up(world: &dyn WorldView, from: BlockPos, ox: i32, oz: i32, back: i32) -> bool {
    let step = |v: i32| v.signum() * back;
    // A run-up along either axis of a diagonal jump still builds speed, so
    // accept the diagonal behind us or either of its cardinal halves.
    let behind = [(-step(ox), -step(oz)), (-step(ox), 0), (0, -step(oz))];
    behind
        .iter()
        .filter(|(bx, bz)| (*bx, *bz) != (0, 0))
        .any(|(bx, bz)| (-1..=1).any(|dy| dry_standable(world, offset(from, *bx, dy, *bz))))
}

/// Whether a body part can occupy `pos`. Lava is walkable-into elsewhere, but
/// never somewhere to aim a jump through. Water is excluded for a different
/// reason: an arc that clips the surface is caught by the water's drag and
/// lands short of where the plan says, so flooded gaps are swum, not jumped.
fn passable(world: &dyn WorldView, pos: BlockPos) -> bool {
    open(world, pos)
}

/// Swim through water, wade into it from land, and climb back out onto a bank.
///
/// A player in water neither stands nor falls: forward motion is about half of
/// walking speed, holding jump drifts upward and releasing it sinks. Every edge
/// here is one block for that reason, because a drift is the only correction
/// available and one block is as far as it can be trusted to carry.
pub struct SwimMove;

impl Move for SwimMove {
    fn candidates(
        &self,
        from: BlockPos,
        world: &dyn WorldView,
        ctx: &MoveContext,
        out: &mut Vec<Edge>,
    ) {
        // `Forbidden` is a one-way door, exactly like the lava buffer: dry land
        // may never step into water, but a bot already in water may move
        // however it needs to in order to get out. Refusing wet-to-wet moves
        // as well would strand a bot in the middle of a pool, with no bank
        // within one stroke of it, which is precisely where the live bots keep
        // respawning.
        //
        // Wet destinations still carry the toll under `Forbidden`, so the
        // search leaves by the shortest route it can find rather than touring
        // the pool.
        let forbidden = ctx.water_policy == WaterPolicy::Forbidden;
        let toll = ctx.water_penalty;
        if on_climbable(world, from) {
            // Letting go of a ladder into water is a drop, not a stroke, and
            // FallMove already prices drops.
            return;
        }
        if !in_water(world, from) {
            if forbidden {
                // Never wade in. Every remaining edge below starts in water, so
                // there is nothing else to offer from dry land.
                return;
            }
            // Wading in from dry land: level with the shore, or one block down
            // to a surface below it. Anything deeper is a fall, and FallMove
            // already plans those, correctly treating water as a safe landing
            // because vanilla cancels all fall damage on entering it.
            for (dx, dz) in CARDINALS {
                for dy in [0, -1] {
                    let to = offset(from, dx, dy, dz);
                    if in_water(world, to) && world.standable(to) {
                        out.push(Edge {
                            to,
                            kind: MoveKind::Swim,
                            cost: ctx.costs.swim.saturating_add(toll),
                        });
                    }
                }
            }
            // Water a block above the feet is not entered. Vanilla would need a
            // jump into the side of a pool and a grab at the surface on the way
            // past, which is a timing the planner cannot promise.
            return;
        }

        // Horizontal strokes, including straight out onto a shore level with
        // the water. Diagonals are left out: two strokes reach the same block,
        // and a diagonal swim would need the corner-clearance rule walking uses
        // without the ground contact that makes walking predictable.
        for (dx, dz) in CARDINALS {
            let to = offset(from, dx, 0, dz);
            if !world.standable(to) {
                continue;
            }
            // Leaving the water sideways onto a dry shore is free under either
            // policy; moving to another water block pays the toll.
            let wet = in_water(world, to);
            out.push(Edge {
                to,
                kind: MoveKind::Swim,
                cost: if wet {
                    ctx.costs.swim.saturating_add(toll)
                } else {
                    ctx.costs.swim
                },
            });
        }

        // Up and down the column. Both are one input away, and rising is what
        // gets a submerged bot back to air before it drowns. Down is kept even
        // under `Forbidden`, because the way out of a pool with an overhang is
        // sometimes underneath it.
        for dy in [1, -1] {
            let to = offset(from, 0, dy, 0);
            if in_water(world, to) && world.standable(to) {
                out.push(Edge {
                    to,
                    kind: MoveKind::Swim,
                    cost: ctx.costs.swim.saturating_add(toll),
                });
            }
        }

        // Climbing out onto a bank one block above the surface. Vanilla does
        // this by drifting up on the jump key and stepping across, so the body
        // needs somewhere to rise into first. A two block bank is not climbable
        // from water at all, which is why only a single block rise is offered:
        // planning the taller one is what leaves a bot circling a pool forever.
        let rise_room = |p: BlockPos| open(world, p) || world.block(p) == BlockKind::Water;
        if !rise_room(offset(from, 0, 1, 0)) || !rise_room(offset(from, 0, 2, 0)) {
            return;
        }
        for (dx, dz) in CARDINALS {
            let to = offset(from, dx, 1, dz);
            if dry_standable(world, to) {
                out.push(Edge {
                    to,
                    kind: MoveKind::Swim,
                    cost: ctx.costs.swim_exit,
                });
            }
        }
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }

    fn metadata(
        &self,
        from: BlockPos,
        edge: &Edge,
        world: &dyn WorldView,
        ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        let primitive = if in_water(world, from) && !in_water(world, edge.to) && edge.to.y > from.y
        {
            crate::PrimitiveId::SWIM_EXIT
        } else {
            crate::PrimitiveId::SWIM
        };
        Some(crate::planning::metadata_from_generated_edge(
            primitive, from, edge.to, edge.kind, world, ctx, edge.cost,
        ))
    }
}

/// The tallest drop vanilla charges nothing for.
pub const SAFE_FALL: i32 = 3;

/// Largest drop inspected by a built-in movement rule.
///
/// Descending parkour stores its signed rise in an `i8`, so 127 is the largest
/// symmetric downward magnitude it can represent. Sharing this bound with
/// ordinary falls also prevents a hostile public [`MoveContext::max_fall`]
/// value from turning one candidate call into billions of world reads.
pub const MAX_GENERATED_DROP: i32 = i8::MAX as i32;

fn generated_drop_limit(configured: i32) -> i32 {
    configured.clamp(0, MAX_GENERATED_DROP)
}

/// Walk off an edge and drop up to `ctx.max_fall` blocks.
///
/// Water counts as a landing: vanilla cancels fall damage outright when the
/// feet enter it, whatever the depth.
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
            if !open(world, step) || !open(world, offset(step, 0, 1, 0)) || world.standable(step) {
                continue;
            }
            // Stop at the first valid landing or obstruction.
            for drop in 1..=generated_drop_limit(ctx.max_fall) {
                let feet = offset(step, 0, -drop, 0);
                // A pool is a landing vanilla is happy with, but dropping into
                // one is still entering water, and under `Forbidden` that is
                // the thing being avoided. Stop the search here rather than
                // continuing through the water to the floor underneath: the
                // bot would never reach it, it would float.
                if ctx.water_policy == WaterPolicy::Forbidden && in_water(world, feet) {
                    break;
                }
                // Falling into a ladder catches the body in vanilla, but it
                // leaves the bot hanging where the follower expected ground.
                // A ladder is entered on purpose, by ClimbMove, or not at all.
                if on_climbable(world, feet) {
                    break;
                }
                if world.standable(feet) {
                    // Water cancels fall damage outright, however deep the drop.
                    let hurt = if in_water(world, feet) {
                        0
                    } else {
                        (drop - SAFE_FALL).max(0) as Cost
                    };
                    out.push(Edge {
                        to: feet,
                        kind: MoveKind::Fall,
                        cost: ctx
                            .costs
                            .fall_base
                            .saturating_add(ctx.costs.fall_per_block.saturating_mul(drop as Cost))
                            .saturating_add(ctx.fall_damage_penalty.saturating_mul(hurt)),
                    });
                    break;
                }
                if !open(world, feet) {
                    break; // hit something unlandable before finding floor
                }
            }
        }
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }

    fn metadata(
        &self,
        from: BlockPos,
        edge: &Edge,
        world: &dyn WorldView,
        ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        Some(crate::planning::metadata_from_generated_edge(
            crate::PrimitiveId::FALL,
            from,
            edge.to,
            edge.kind,
            world,
            ctx,
            edge.cost,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct MetadataGrid(HashMap<(i32, i32, i32), BlockKind>);

    impl WorldView for MetadataGrid {
        fn block(&self, pos: BlockPos) -> BlockKind {
            self.0
                .get(&(pos.x, pos.y, pos.z))
                .copied()
                .unwrap_or(BlockKind::Air)
        }
    }

    fn generated_metadata(
        movement: &dyn Move,
        from: BlockPos,
        to: BlockPos,
        world: &dyn WorldView,
        context: &MoveContext,
    ) -> (Cost, crate::planning::MotionMetadata) {
        let mut edges = Vec::new();
        movement.candidates(from, world, context, &mut edges);
        let edge = edges
            .iter()
            .find(|edge| edge.to == to)
            .unwrap_or_else(|| panic!("no generated edge from {from:?} to {to:?}"));
        let metadata = movement
            .metadata(from, edge, world, context)
            .expect("built-in edge omitted metadata");
        assert_eq!(
            metadata.predicted.total(),
            edge.cost,
            "metadata diverged for {:?} from {from:?} to {to:?}",
            edge.kind
        );
        (edge.cost, metadata)
    }

    #[test]
    fn representative_generated_edges_preserve_exact_cost_components() {
        let context = MoveContext::default();

        let step_from = BlockPos::new(0, 64, 0);
        let step_to = BlockPos::new(1, 65, 0);
        let step_world = MetadataGrid(
            [
                ((0, 64, 0), BlockKind::Step(8)),
                ((1, 64, 0), BlockKind::Solid),
            ]
            .into_iter()
            .collect(),
        );
        let (cost, metadata) =
            generated_metadata(&WalkMove, step_from, step_to, &step_world, &context);
        assert_eq!(cost, context.costs.step);
        assert_eq!(metadata.primitive, crate::PrimitiveId::STEP);

        let jump_from = BlockPos::new(0, 64, 0);
        let jump_to = BlockPos::new(1, 65, 0);
        let jump_world = MetadataGrid([((1, 64, 0), BlockKind::Solid)].into_iter().collect());
        let (cost, metadata) =
            generated_metadata(&JumpMove, jump_from, jump_to, &jump_world, &context);
        assert_eq!(cost, context.costs.jump);
        assert_eq!(metadata.primitive, crate::PrimitiveId::JUMP);

        let climb_from = BlockPos::new(0, 64, 0);
        let climb_to = BlockPos::new(1, 65, 0);
        let climb_world = MetadataGrid([((1, 65, 0), BlockKind::Climbable)].into_iter().collect());
        let (cost, metadata) =
            generated_metadata(&ClimbMove, climb_from, climb_to, &climb_world, &context);
        assert_eq!(
            cost,
            context
                .costs
                .climb
                .saturating_mul(2)
                .saturating_add(context.costs.jump)
        );
        assert_eq!(metadata.predicted.time, cost);
        assert_eq!(metadata.predicted_ticks, 41);
        assert_eq!(metadata.features.horizontal_blocks, 1);

        let rung_from = BlockPos::new(0, 64, 0);
        let rung_to = BlockPos::new(0, 65, 0);
        let rung_world = MetadataGrid(
            [
                ((0, 64, 0), BlockKind::Climbable),
                ((0, 65, 0), BlockKind::Climbable),
            ]
            .into_iter()
            .collect(),
        );
        let (cost, metadata) =
            generated_metadata(&ClimbMove, rung_from, rung_to, &rung_world, &context);
        assert_eq!(cost, context.costs.climb);
        assert_eq!(metadata.predicted_ticks, 15);
        assert_eq!(metadata.features.horizontal_blocks, 0);

        let parkour_from = BlockPos::new(0, 64, 0);
        let parkour_to = BlockPos::new(3, 64, 0);
        let parkour_world = MetadataGrid(
            [
                ((-1, 63, 0), BlockKind::Solid),
                ((0, 63, 0), BlockKind::Solid),
                ((3, 63, 0), BlockKind::Solid),
            ]
            .into_iter()
            .collect(),
        );
        let (cost, metadata) = generated_metadata(
            &ParkourMove,
            parkour_from,
            parkour_to,
            &parkour_world,
            &context,
        );
        assert_eq!(cost, context.costs.parkour_per_block.saturating_mul(3));
        assert_eq!(metadata.primitive, crate::PrimitiveId::PARKOUR);
    }

    #[test]
    fn swim_toll_is_fixed_safety_under_both_policies() {
        let from = BlockPos::new(0, 64, 0);
        let to = BlockPos::new(1, 64, 0);
        let world = MetadataGrid(
            [
                ((0, 64, 0), BlockKind::Water),
                ((1, 64, 0), BlockKind::Water),
            ]
            .into_iter()
            .collect(),
        );
        for policy in [WaterPolicy::Forbidden, WaterPolicy::Penalized] {
            let context = MoveContext {
                water_policy: policy,
                water_penalty: 777,
                ..MoveContext::default()
            };
            let (cost, metadata) = generated_metadata(&SwimMove, from, to, &world, &context);
            assert_eq!(
                cost,
                context.costs.swim.saturating_add(context.water_penalty)
            );
            assert_eq!(metadata.predicted.time, context.costs.swim);
            assert_eq!(metadata.predicted.safety, context.water_penalty);
            assert_eq!(metadata.predicted.damage, 0);
            assert_eq!(
                metadata.features.flags & crate::MoveFeatures::FLAG_WATER_ENTRY,
                0
            );
        }

        let context = MoveContext {
            water_policy: WaterPolicy::Penalized,
            water_penalty: 777,
            ..MoveContext::default()
        };
        let entry_from = BlockPos::new(-1, 64, 0);
        let entry_world = MetadataGrid([((0, 64, 0), BlockKind::Water)].into_iter().collect());
        let (entry_cost, entry) =
            generated_metadata(&SwimMove, entry_from, from, &entry_world, &context);
        let (stroke_cost, stroke) = generated_metadata(&SwimMove, from, to, &world, &context);
        assert_eq!(entry_cost, stroke_cost);
        assert_eq!(entry.predicted, stroke.predicted);
        assert_ne!(
            entry.features.flags & crate::MoveFeatures::FLAG_WATER_ENTRY,
            0
        );
        assert_eq!(
            stroke.features.flags & crate::MoveFeatures::FLAG_WATER_ENTRY,
            0
        );
        assert_ne!(entry.features, stroke.features);
    }

    #[test]
    fn fall_metadata_matches_dry_damage_and_zeroes_water_landing_damage() {
        let from = BlockPos::new(0, 70, 0);
        let to = BlockPos::new(1, 65, 0);
        let context = MoveContext {
            water_policy: WaterPolicy::Penalized,
            ..MoveContext::default()
        };

        let dry_world = MetadataGrid([((1, 64, 0), BlockKind::Solid)].into_iter().collect());
        let (dry_cost, dry) = generated_metadata(&FallMove, from, to, &dry_world, &context);
        assert_eq!(
            dry.predicted.damage,
            context.fall_damage_penalty.saturating_mul(2)
        );
        assert_eq!(dry.predicted.total(), dry_cost);

        let water_world = MetadataGrid([((1, 65, 0), BlockKind::Water)].into_iter().collect());
        let (water_cost, water) = generated_metadata(&FallMove, from, to, &water_world, &context);
        assert_eq!(water.predicted.damage, 0);
        assert_eq!(
            water_cost,
            context
                .costs
                .fall_base
                .saturating_add(context.costs.fall_per_block.saturating_mul(5))
        );
        assert_eq!(water.predicted.total(), water_cost);
    }

    #[test]
    fn forbidden_policy_only_allows_risk_reducing_escape_steps() {
        let policy = LavaPolicy::Forbidden { clearance: 2 };

        assert!(lava_transition_allowed(0, 0, policy));
        assert!(!lava_transition_allowed(0, 1, policy));
        assert!(lava_transition_allowed(3, 2, policy));
        assert!(!lava_transition_allowed(3, 3, policy));
        assert!(!lava_transition_allowed(2, 3, policy));
    }
    /// Both ways onto a ladder have to exist, and stepping in must be cheaper.
    ///
    /// A bot walking up to the foot of a ladder should walk into it. It was
    /// instead planning to grab the rung above - which means rising and moving
    /// sideways on the same tick, into a block whose collision box is a sliver
    /// against the far wall - and wedging itself on that sliver.
    #[test]
    fn a_ladder_is_entered_at_foot_level_by_preference() {
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

        let mut blocks = HashMap::new();
        // Floor to stand on, and a wall with a ladder up its face.
        for x in 0..=2 {
            for z in 0..=2 {
                blocks.insert((x, 63, z), BlockKind::Solid);
            }
        }
        for y in 64..=67 {
            blocks.insert((2, y, 1), BlockKind::Solid);
            blocks.insert((1, y, 1), BlockKind::Climbable);
        }
        let world = Grid(blocks);

        let mut edges = Vec::new();
        ClimbMove.candidates(
            BlockPos::new(1, 64, 0),
            &world,
            &MoveContext::default(),
            &mut edges,
        );

        let foot = edges.iter().find(|e| e.to == BlockPos::new(1, 64, 1));
        let reach = edges.iter().find(|e| e.to == BlockPos::new(1, 65, 1));
        let foot = foot.expect("no way onto the ladder at foot level");
        let reach = reach.expect("no way onto the ladder one rung up");
        assert!(
            foot.cost < reach.cost,
            "stepping in ({}) is not cheaper than reaching up ({})",
            foot.cost,
            reach.cost
        );
    }

    #[test]
    fn descending_parkour_finds_a_landing_below_the_first_drop() {
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

        let world = Grid(
            [
                ((-1, 63, 0), BlockKind::Solid),
                ((0, 63, 0), BlockKind::Solid),
                ((3, 60, 0), BlockKind::Solid),
            ]
            .into_iter()
            .collect(),
        );
        let mut edges = Vec::new();
        ParkourMove.candidates(
            BlockPos::new(0, 64, 0),
            &world,
            &MoveContext::default(),
            &mut edges,
        );

        assert!(
            edges.iter().any(|edge| edge.to == BlockPos::new(3, 61, 0)),
            "three-block-lower landing was not generated"
        );
    }

    #[test]
    fn extreme_max_fall_is_bounded_and_parkour_rise_never_wraps() {
        let from = BlockPos::new(0, 200, 0);
        let parkour_to = BlockPos::new(2, 200 - MAX_GENERATED_DROP, 0);
        let parkour_world = MetadataGrid(
            [
                ((0, 199, 0), BlockKind::Solid),
                ((2, parkour_to.y - 1, 0), BlockKind::Solid),
            ]
            .into_iter()
            .collect(),
        );
        let fall_to = BlockPos::new(1, 200 - MAX_GENERATED_DROP, 0);
        let fall_world = MetadataGrid(
            [
                ((0, 199, 0), BlockKind::Solid),
                ((1, fall_to.y - 1, 0), BlockKind::Solid),
            ]
            .into_iter()
            .collect(),
        );

        assert_eq!(generated_drop_limit(128), MAX_GENERATED_DROP);
        assert_eq!(generated_drop_limit(i32::MAX), MAX_GENERATED_DROP);
        for max_fall in [128, i32::MAX] {
            let context = MoveContext {
                max_fall,
                ..MoveContext::default()
            };

            let mut parkour = Vec::new();
            ParkourMove.candidates(from, &parkour_world, &context, &mut parkour);
            let edge = parkour
                .iter()
                .find(|edge| edge.to == parkour_to)
                .expect("representable deepest parkour landing was omitted");
            let MoveKind::Parkour { rise, .. } = edge.kind else {
                panic!("deep parkour landing used the wrong move kind");
            };
            assert_eq!(rise, -127);

            let mut falls = Vec::new();
            FallMove.candidates(from, &fall_world, &context, &mut falls);
            assert!(
                falls.iter().any(|edge| edge.to == fall_to),
                "representable deepest fall landing was omitted"
            );
            assert!(
                parkour
                    .iter()
                    .chain(&falls)
                    .all(|edge| { from.y.saturating_sub(edge.to.y) <= MAX_GENERATED_DROP })
            );
        }

        // A landing one block past the encoding/work cap must not be emitted,
        // even when the caller asks for that distance or an effectively
        // unbounded one.
        let too_deep_parkour = BlockPos::new(2, from.y - (MAX_GENERATED_DROP + 1), 0);
        let too_deep_fall = BlockPos::new(1, from.y - (MAX_GENERATED_DROP + 1), 0);
        let parkour_world = MetadataGrid(
            [
                ((0, 199, 0), BlockKind::Solid),
                ((2, too_deep_parkour.y - 1, 0), BlockKind::Solid),
            ]
            .into_iter()
            .collect(),
        );
        let fall_world = MetadataGrid(
            [
                ((0, 199, 0), BlockKind::Solid),
                ((1, too_deep_fall.y - 1, 0), BlockKind::Solid),
            ]
            .into_iter()
            .collect(),
        );
        let context = MoveContext {
            max_fall: i32::MAX,
            ..MoveContext::default()
        };
        let mut parkour = Vec::new();
        ParkourMove.candidates(from, &parkour_world, &context, &mut parkour);
        assert!(!parkour.iter().any(|edge| edge.to == too_deep_parkour));
        let mut falls = Vec::new();
        FallMove.candidates(from, &fall_world, &context, &mut falls);
        assert!(!falls.iter().any(|edge| edge.to == too_deep_fall));
    }

    /// The wall toll scales with room: a corridor is charged far less than an
    /// open plaza, because in a corridor there is no way to be more central.
    #[test]
    fn wall_toll_fades_in_a_corridor() {
        use std::collections::HashMap;
        struct Grid(HashMap<(i32, i32, i32), BlockKind>);
        impl WorldView for Grid {
            fn block(&self, p: BlockPos) -> BlockKind {
                self.0
                    .get(&(p.x, p.y, p.z))
                    .copied()
                    .unwrap_or(BlockKind::Air)
            }
        }
        // A one-wide east-west corridor at y=64: floor at 63, walls at z=+/-1.
        let mut corridor = HashMap::new();
        for x in -3..=3 {
            corridor.insert((x, 63, 0), BlockKind::Solid); // floor
            for h in 0..=1 {
                corridor.insert((x, 64 + h, 1), BlockKind::Solid); // wall
                corridor.insert((x, 64 + h, -1), BlockKind::Solid); // wall
            }
        }
        let corridor = Grid(corridor);

        // An open plaza: a floor with a single wall block to the north, so the
        // node is wall-adjacent but has room on every other side.
        let mut plaza = HashMap::new();
        for x in -3..=3 {
            for z in -3..=3 {
                plaza.insert((x, 63, z), BlockKind::Solid);
            }
        }
        plaza.insert((0, 64, 1), BlockKind::Solid);
        plaza.insert((0, 65, 1), BlockKind::Solid);
        let plaza = Grid(plaza);

        let here = BlockPos::new(0, 64, 0);
        let in_corridor = wall_proximity_penalty(here, &corridor, 6);
        let in_open = wall_proximity_penalty(here, &plaza, 6);

        assert!(in_corridor > 0, "corridor should still cost a little");
        assert!(
            in_open > in_corridor,
            "open plaza ({in_open}) should be charged more than a corridor ({in_corridor})"
        );
        assert_eq!(
            wall_proximity_penalty(here, &plaza, Cost::MAX),
            Cost::MAX,
            "an extreme public wall toll must saturate after scaling"
        );
    }
}
