//! Azalea plugin and navigation API.

use std::future::Future;

use azalea::StartSprintEvent;
use azalea::StartWalkEvent;
use azalea::app::{App, Plugin};
use azalea::bot::JumpEvent;
use azalea::ecs::prelude::*;
use azalea::entity::{LocalEntity, LookDirection, Physics, Position};
use azalea::local_player::{Hunger, WorldHolder};
use azalea::physics::PhysicsSystems;
use azalea::prelude::GameTick;
use azalea::{BlockPos, Client, SprintDirection, WalkDirection};
use bevy_tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use crate::{
    FollowerDirective, FollowerFrame, FollowerSettings, MoveContext, PathFollower, WorldSnapshot,
    WorldView,
    default_moves, find_path_best_effort, steering_direction,
};

#[derive(Debug, Clone, Resource)]
pub struct PathfinderSettings {
    /// Ticks between fallback replans for dynamic goals. Set to 0 to disable.
    pub dynamic_replan_ticks: u32,
    pub max_plan_legs: u32,
    pub snapshot_margin: i32,
    pub snapshot_max_span_xz: i32,
    pub snapshot_max_span_y: i32,
    pub planner: MoveContext,
    pub follower: FollowerSettings,
}

impl Default for PathfinderSettings {
    fn default() -> Self {
        Self {
            dynamic_replan_ticks: 0,
            max_plan_legs: 16,
            // The snapshot is the *only* world the planner ever sees, so this
            // margin is not cosmetic slack: it is how far a route is allowed to
            // detour off the straight start->goal line. At 10 the box collapses
            // to a 21 block corridor whenever start and goal happen to line up
            // on an axis, and any route that has to go round a building or a
            // ridge is invisible. Measured on the hub: village (268,31,187) ->
            // coal mine (228,23,127) needs to swing out to x=212, six blocks
            // outside the old box, so it failed; at 48 it plans in one leg.
            snapshot_margin: 48,
            snapshot_max_span_xz: 128,
            // Tall enough to hold a climb. A hand-built map spans well over 64
            // blocks of height, and when the goal is above the span cap the
            // goal itself falls outside the snapshot, so the search degenerates
            // into greedy descent on straight-line distance and walks the bot
            // into the first cave or dead-end bridge that happens to be higher.
            //
            // 256 rather than 160 so that a climb from sea level to a summit
            // holds both ends at once: a world is only about 384 blocks tall,
            // and it is the XZ cap that bounds the volume, so height here is
            // nearly free. At 160 the hub summit fell outside its own snapshot
            // on every approach from low ground.
            snapshot_max_span_y: 256,
            planner: MoveContext::default(),
            follower: FollowerSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationGoal {
    Fixed(BlockPos),
    Dynamic { position: BlockPos, revision: u64 },
}

impl NavigationGoal {
    pub fn position(self) -> BlockPos {
        match self {
            Self::Fixed(position) | Self::Dynamic { position, .. } => position,
        }
    }
}

/// Add this to a local player to start or replace navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct NavigationRequest {
    pub goal: NavigationGoal,
    pub path_seed: u64,
    /// Increase for each request; older results are ignored.
    pub generation: u64,
}

/// Pauses navigation without using the stall budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component, Default)]
pub struct NavigationPaused(pub bool);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component, Default)]
pub enum NavigationStatus {
    #[default]
    Idle,
    Planning {
        generation: u64,
    },
    Following {
        generation: u64,
    },
    Paused {
        generation: u64,
    },
    Arrived {
        generation: u64,
    },
    Failed {
        generation: u64,
    },
}

#[derive(Component)]
struct NavigationTask(Task<PlanResult>);

#[derive(Component)]
struct ActiveNavigation {
    follower: PathFollower,
    reached_goal: bool,
    legs: u32,
    /// Consecutive legs that ended within [`STALL_RADIUS`] of where they began.
    stalled_legs: u32,
    dynamic_ticks: u32,
}

#[derive(Component)]
struct NavigationTerminal;

/// Blocks this navigation has already failed on, and what they now cost.
///
/// Cleared whenever a new [`NavigationRequest`] arrives, because the memory is
/// about one journey: a doorway that was impassable while carrying a path from
/// the north may be the obvious way in from the south.
#[derive(Component, Default)]
struct AvoidMemory {
    spots: std::collections::HashMap<BlockPos, crate::Cost>,
}

/// Toll added to a block each time a leg dies on it, in walk-cost units where
/// one flat block is 10. The first failure is worth an 80-block detour, and
/// repeats escalate, so a spot that keeps failing is eventually routed around
/// however long the way round is.
const AVOID_PENALTY: crate::Cost = 800;

/// Radius of the ball charged around a failed block.
///
/// Charging the single block is not enough: the bot rarely wedges on the exact
/// block the follower reports, and every neighbour offers a path through the
/// same doorway at almost the same cost.
const AVOID_RADIUS: i32 = 2;

impl AvoidMemory {
    fn blame(&mut self, at: BlockPos) {
        for dx in -AVOID_RADIUS..=AVOID_RADIUS {
            for dy in -AVOID_RADIUS..=AVOID_RADIUS {
                for dz in -AVOID_RADIUS..=AVOID_RADIUS {
                    let pos = BlockPos::new(at.x + dx, at.y + dy, at.z + dz);
                    *self.spots.entry(pos).or_insert(0) =
                        self.spots.get(&pos).copied().unwrap_or(0).saturating_add(AVOID_PENALTY);
                }
            }
        }
    }

    fn snapshot(&self) -> std::sync::Arc<std::collections::HashMap<BlockPos, crate::Cost>> {
        std::sync::Arc::new(self.spots.clone())
    }
}

struct PlanResult {
    request: NavigationRequest,
    path: crate::Path,
    reached_goal: bool,
    legs: u32,
    stalled_legs: u32,
    /// Where this leg was planned from, so "did this leg go anywhere" is
    /// answerable without trusting the follower to notice.
    start: BlockPos,
}

/// How far a leg has to travel to count as progress.
///
/// A best-effort plan that stops this close to its own start is not a route,
/// it is the planner saying "nowhere better from here". Following it makes the
/// follower report `Arrived` on the next tick, which replans, which returns the
/// same path, which arrives again: a leg burned per tick with the bot standing
/// still. Two blocks is just outside the follower's own `close_enough_xz`.
const STALL_RADIUS: i32 = 3;

/// How many stalled legs to spend before giving up.
///
/// Each retry re-seeds the tie-break, so these are genuinely different attempts
/// rather than the same deterministic search run again.
const MAX_STALLED_LEGS: u32 = 6;

/// Plans and follows block paths for local players.
pub struct AzaleaPathfinderPlugin;

impl Plugin for AzaleaPathfinderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PathfinderSettings>()
            .add_systems(azalea::app::PreUpdate, add_navigation_status)
            .add_systems(
                azalea::app::Update,
                (
                    start_requested_navigation,
                    poll_navigation_tasks,
                    stop_removed_navigation,
                )
                    .chain()
                    // These emit StartWalkEvent, so they have to run before the
                    // systems that drain it, or a stop lands a frame late and
                    // the bot walks one tick further than it planned to.
                    .before(azalea::movement::MoveEventsSystems)
                    // Same reason as the GameTick ordering below: azalea's own
                    // pathfinder writes the same events from this schedule.
                    .after(azalea::pathfinder::PathfinderSystems),
            )
            // send_position must have already built this tick's movement packet
            // before the follower touches Physics or LookDirection. Without
            // this the two systems have no ordering relative to each other, so
            // on some ticks the packet is built from a state the follower has
            // already changed, and a movement anticheat sees a position that
            // does not match the inputs. Azalea's own path executor orders
            // itself the same way.
            .add_systems(
                GameTick,
                tick_navigation
                    .in_set(NavigationSystems)
                    .after(PhysicsSystems)
                    .after(azalea::movement::send_position)
                    // Azalea ships its own pathfinder, whose executor writes the
                    // same walk/sprint/jump events. Both are registered even
                    // when only one is driving, so pin the order rather than
                    // leave it to chance.
                    .after(azalea::pathfinder::PathfinderSystems),
            );
    }
}

/// The set the follower runs in.
///
/// Anything else that steers the bot writes the same walk, sprint and jump
/// events, so it has to be ordered against this set or bevy picks an order per
/// run and the two take turns winning.
#[derive(Clone, Debug, Eq, Hash, PartialEq, SystemSet)]
pub struct NavigationSystems;

fn add_navigation_status(
    mut commands: Commands,
    query: Query<Entity, (With<LocalEntity>, Without<NavigationStatus>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(NavigationStatus::Idle);
    }
}

#[allow(clippy::type_complexity)]
fn start_requested_navigation(
    mut commands: Commands,
    settings: Res<PathfinderSettings>,
    query: Query<(
        Entity,
        &NavigationRequest,
        Ref<NavigationRequest>,
        &Position,
        &WorldHolder,
        Option<&NavigationTask>,
        Option<&ActiveNavigation>,
        Option<&NavigationTerminal>,
    )>,
    mut walk_events: MessageWriter<StartWalkEvent>,
) {
    for (entity, request, request_ref, position, world, task, active, terminal) in &query {
        let changed = request_ref.is_changed();
        if !changed && (task.is_some() || active.is_some() || terminal.is_some()) {
            continue;
        }
        if changed {
            commands
                .entity(entity)
                .remove::<(ActiveNavigation, NavigationTask, NavigationTerminal)>();
        }
        // Stop old movement while the replacement path is planned.
        stop(entity, &mut walk_events);
        let task = spawn_plan(
            *request,
            BlockPos::from(&**position),
            world.shared.clone(),
            settings.clone(),
            1,
            0,
            std::sync::Arc::default(),
        );
        commands.entity(entity).insert((
            NavigationTask(task),
            NavigationStatus::Planning {
                generation: request.generation,
            },
            // A new destination starts with a clean slate.
            AvoidMemory::default(),
        ));
    }
}

/// `PF_NAV_DEBUG=1` traces every leg and every reason a leg ended.
///
/// Without it a stuck bot is indistinguishable from a walking one at any
/// sampling rate a shell script can manage: the status flickers
/// Planning/Following several times a second, so polling shows "Following"
/// forever while the bot re-walks the same two metres.
fn nav_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PF_NAV_DEBUG").is_ok())
}

fn spawn_plan(
    request: NavigationRequest,
    start: BlockPos,
    world: std::sync::Arc<parking_lot::RwLock<azalea::world::World>>,
    settings: PathfinderSettings,
    legs: u32,
    stalled_legs: u32,
    avoid: std::sync::Arc<std::collections::HashMap<BlockPos, crate::Cost>>,
) -> Task<PlanResult> {
    if nav_debug() {
        eprintln!("nav plan leg {legs} from {start:?} with {} tolled blocks", avoid.len());
    }
    AsyncComputeTaskPool::get().spawn(async move {
        let target = request.goal.position();
        let (lo, hi) = snapshot_bounds(start, target, &settings);
        let snapshot = WorldSnapshot::capture(&world, lo, hi);
        let mut context = settings.planner.clone();
        context.avoid = avoid;
        // Re-seed per leg. The search is deterministic, so replanning from the
        // same block with the same seed returns the identical path, and a bot
        // wedged on a corner wedges on it again for every leg it has left.
        // Varying the tie-break is what makes a retry a retry.
        context.path_seed = request
            .path_seed
            .wrapping_add((legs as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        // A goal block that cannot be stood in can never be "reached", so the
        // plan is marked failed even when the bot walks the entire route and
        // ends up on top of it. That happens constantly with hand-typed or
        // clicked coordinates, where the Y is a guess. Snap to the nearest
        // standable block first so success means what a person means by it.
        let target = snap_to_standable(&snapshot, target, lo, hi).unwrap_or(target);
        let (path, reached_goal) =
            find_path_best_effort(&snapshot, start, target, &default_moves(), &context);
        PlanResult {
            request,
            path,
            reached_goal,
            legs,
            stalled_legs,
            start,
        }
    })
}

/// Nearest block to `goal` the bot could actually stand in.
///
/// Searched nearest-first and vertically before horizontally: a goal whose Y is
/// wrong is nearly always a column that is right, so the floor below or the air
/// above is what was meant. The radius is deliberately small, because snapping
/// far enough to reach a *different* place would be worse than failing.
fn snap_to_standable(world: &WorldSnapshot, goal: BlockPos, lo: BlockPos, hi: BlockPos) -> Option<BlockPos> {
    // A goal outside the snapshot is a goal we know nothing about, and snapping
    // it lands somewhere the box happens to end rather than somewhere the world
    // does. Climbing the hub mountain from sea level hit this exactly: the
    // summit sat above the snapshot's Y ceiling, so the goal was relocated 88
    // blocks down onto the highest ledge still inside the box, the search
    // honestly reached it, and the bot reported Arrived while standing
    // three quarters of the way up. Leave a goal we cannot see where it is and
    // let the next leg, planned from higher up, see it properly.
    if goal.y < lo.y || goal.y > hi.y || goal.x < lo.x || goal.x > hi.x || goal.z < lo.z || goal.z > hi.z
    {
        return None;
    }
    const RADIUS: i32 = 4;
    /// Vertical reach, which is deliberately far larger than the horizontal.
    ///
    /// A goal's X and Z come from a map or a click and are what the person
    /// meant; its Y is a guess, and on this testbed it is routinely a hundred
    /// blocks of open sky above the ground. Four blocks of vertical search
    /// could never find the floor under such a goal, so the goal stayed
    /// unreachable and the search fell back to shrinking straight-line
    /// distance, in which the Y error dominates: the bot would climb the
    /// nearest mountain rather than walk to the X and Z it was given.
    const RADIUS_Y: i32 = 192;
    if world.standable(goal) {
        return Some(goal);
    }
    let mut best: Option<(i32, BlockPos)> = None;
    for dy in -RADIUS_Y..=RADIUS_Y {
        for dx in -RADIUS..=RADIUS {
            for dz in -RADIUS..=RADIUS {
                let candidate = BlockPos::new(goal.x + dx, goal.y + dy, goal.z + dz);
                if !world.standable(candidate) {
                    continue;
                }
                // Horizontal error moves the bot somewhere else entirely, so
                // weight it above vertical error rather than using distance.
                let score = (dx.abs() + dz.abs()) * 4 + dy.abs();
                if best.is_none_or(|(best_score, _)| score < best_score) {
                    best = Some((score, candidate));
                }
            }
        }
    }
    best.map(|(_, pos)| pos)
}

/// Directory the path visualiser reads, if set. One file per bot.
fn path_viz_dir() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var("PF_PATH_DIR").ok()).as_deref()
}

/// Write one bot's planned path where a server-side script can draw it.
///
/// The format is deliberately trivial - a name, the goal, and a flat list of
/// `x y z kind` lines - so the reader is a shell loop and a `/particle` command
/// rather than anything that has to parse. A move kind per node is what lets the
/// drawing colour a jump differently from a walk, which is the whole point:
/// you can watch the bot decide to parkour a gap before it reaches it.
fn write_path_viz(
    profile: Option<&azalea::player::GameProfileComponent>,
    goal: BlockPos,
    path: &crate::Path,
) {
    let Some(dir) = path_viz_dir() else { return };
    let Some(profile) = profile else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let mut out = format!("{}\n{} {} {}\n", profile.name, goal.x, goal.y, goal.z);
    for node in &path.nodes {
        use std::fmt::Write;
        let kind = match node.reached_by {
            crate::MoveKind::Start => "start",
            crate::MoveKind::Walk => "walk",
            crate::MoveKind::Jump => "jump",
            crate::MoveKind::Parkour { .. } => "parkour",
            crate::MoveKind::Climb => "climb",
            crate::MoveKind::Fall => "fall",
            crate::MoveKind::Swim => "swim",
            crate::MoveKind::Aotv | crate::MoveKind::Etherwarp => "warp",
        };
        let _ = writeln!(out, "{} {} {} {}", node.pos.x, node.pos.y, node.pos.z, kind);
    }
    let _ = std::fs::write(format!("{dir}/{}.path", profile.name), out);
}

#[allow(clippy::type_complexity)]
fn poll_navigation_tasks(
    mut commands: Commands,
    settings: Res<PathfinderSettings>,
    mut query: Query<(
        Entity,
        &NavigationRequest,
        &mut NavigationTask,
        &WorldHolder,
        Option<&AvoidMemory>,
        &Physics,
        Option<&azalea::player::GameProfileComponent>,
    )>,
) {
    for (entity, current, mut task, world_holder, avoid, physics, profile) in &mut query {
        let avoid = avoid.map(AvoidMemory::snapshot).unwrap_or_default();
        let Some(result) = future::block_on(future::poll_once(&mut task.0)) else {
            continue;
        };
        commands.entity(entity).remove::<NavigationTask>();
        if *current != result.request {
            continue;
        }
        if result.reached_goal && result.path.nodes.len() < 2 {
            commands.entity(entity).insert((
                NavigationStatus::Arrived {
                    generation: current.generation,
                },
                NavigationTerminal,
            ));
            continue;
        }
        // Did this leg actually go anywhere? A best-effort plan that ends on
        // top of its own start is the planner reporting a dead end, whether it
        // says so with one node or with four. Following it is what produced the
        // Planning/Following flicker: instant arrival, instant replan, no
        // movement, until the leg budget ran out a minute later.
        let end = result
            .path
            .nodes
            .last()
            .map_or(result.start, |node| node.pos);
        let travelled = (end.x - result.start.x).abs()
            + (end.y - result.start.y).abs()
            + (end.z - result.start.z).abs();
        if !result.reached_goal && travelled < STALL_RADIUS {
            // A block position taken mid-fall is not a place the planner can
            // reason about: the feet are in air with air underneath, so no move
            // applies and the plan comes back empty however good the route is.
            // Falling off the mountain's west face burned the entire stall
            // budget in under a second this way and reported Failed while the
            // bot was still in the air. Retrying without spending budget lets
            // the bot land and plan from somewhere real.
            //
            // A ladder is not mid-fall, though. `on_ground` is false for the
            // whole of a climb, so a bot on one was exempt from the budget
            // forever: it climbed, walked out of the shaft, fell, replanned and
            // climbed again, and a single navigation ran for over ten minutes
            // without ever reporting Failed. The exemption is for a body in
            // flight with nothing under it, which is a state that ends by
            // itself in under a second; hanging on a ladder is somewhere the
            // planner can reason about perfectly well.
            let on_ladder = {
                use crate::local::world::WorldView;
                let world = world_holder.shared.read();
                world.block(result.start) == crate::local::world::BlockKind::Climbable
            };
            let airborne = !physics.on_ground() && !on_ladder;
            let stalled_legs = if airborne {
                result.stalled_legs
            } else {
                result.stalled_legs.saturating_add(1)
            };
            let legs = if airborne { result.legs } else { result.legs + 1 };
            if !airborne
                && (stalled_legs > MAX_STALLED_LEGS || result.legs >= settings.max_plan_legs.max(1))
            {
                commands.entity(entity).insert((
                    NavigationStatus::Failed {
                        generation: current.generation,
                    },
                    NavigationTerminal,
                ));
            } else {
                // Straight back to the planner with a fresh seed rather than
                // handing the follower a path to nowhere.
                let task = spawn_plan(
                    *current,
                    result.start,
                    world_holder.shared.clone(),
                    settings.clone(),
                    legs,
                    stalled_legs,
                    avoid.clone(),
                );
                commands.entity(entity).insert((
                    NavigationTask(task),
                    NavigationStatus::Planning {
                        generation: current.generation,
                    },
                ));
            }
            continue;
        }
        if nav_debug() {
            let end = result.path.nodes.last().map(|node| node.pos);
            eprintln!(
                "nav leg {} from {:?} -> {} nodes, ends {:?}, reached_goal={}\n  steps: {:?}",
                result.legs,
                result.start,
                result.path.nodes.len(),
                end,
                result.reached_goal,
                result
                    .path
                    .nodes
                    .iter()
                    // The whole route, not a preview. Five nodes was enough to
                    // see where a leg started and never enough to see why it
                    // chose that way, which is the question worth asking.
                    .map(|node| (node.pos, node.reached_by))
                    .collect::<Vec<_>>(),
            );
        }
        // Publish the planned path for the in-game visualiser, if one is
        // watching. Written here, on every (re)plan, so what the player sees
        // ingame is exactly the route the follower is about to walk - and it
        // refreshes the instant the bot changes its mind.
        write_path_viz(profile, current.goal.position(), &result.path);
        let follower = PathFollower::new(result.path, settings.follower.clone(), current.path_seed);
        commands.entity(entity).insert((
            ActiveNavigation {
                follower,
                reached_goal: result.reached_goal,
                legs: result.legs,
                stalled_legs: 0,
                dynamic_ticks: 0,
            },
            NavigationStatus::Following {
                generation: current.generation,
            },
        ));
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn tick_navigation(
    mut commands: Commands,
    settings: Res<PathfinderSettings>,
    mut query: Query<(
        Entity,
        &NavigationRequest,
        Option<&NavigationPaused>,
        &Position,
        &Physics,
        &mut LookDirection,
        &WorldHolder,
        Option<&Hunger>,
        &mut ActiveNavigation,
        Option<&mut AvoidMemory>,
    )>,
    mut walk_events: MessageWriter<StartWalkEvent>,
    mut sprint_events: MessageWriter<StartSprintEvent>,
    mut jump_events: MessageWriter<JumpEvent>,
) {
    for (
        entity,
        request,
        paused,
        position,
        physics,
        mut look,
        world_holder,
        hunger,
        mut active,
        mut avoid,
    ) in &mut query
    {
        let paused = paused.is_some_and(|paused| paused.0);
        if paused {
            let directive = active.follower.tick(
                &UnavailableWorld,
                FollowerFrame {
                    position: **position,
                    on_ground: physics.on_ground(),
                    horizontal_collision: physics.horizontal_collision,
                    paused: true,
                },
            );
            debug_assert_eq!(directive, FollowerDirective::Paused);
            stop(entity, &mut walk_events);
            commands.entity(entity).insert(NavigationStatus::Paused {
                generation: request.generation,
            });
            continue;
        }

        // Check nearby blocks again before following the planned path.
        let (low, high) = follower_snapshot_bounds(
            **position,
            active.follower.path(),
            active.follower.current_node_index(),
            &settings.follower,
        );
        let live_snapshot = WorldSnapshot::capture(&world_holder.shared, low, high);
        let directive = active.follower.tick(
            &live_snapshot,
            FollowerFrame {
                position: **position,
                on_ground: physics.on_ground(),
                horizontal_collision: physics.horizontal_collision,
                paused: false,
            },
        );
        match directive {
            FollowerDirective::Paused => {
                stop(entity, &mut walk_events);
                commands.entity(entity).insert(NavigationStatus::Paused {
                    generation: request.generation,
                });
            }
            FollowerDirective::Move {
                target,
                sprint,
                jump,
                yaw_bias,
                pitch_bias,
                max_turn,
            } => {
                let (yaw, pitch) = steering_direction(
                    **position,
                    look.y_rot(),
                    look.x_rot(),
                    target,
                    yaw_bias,
                    pitch_bias,
                    max_turn,
                    &settings.follower,
                );
                look.update(LookDirection::new(yaw, pitch));
                // Vanilla refuses to start a sprint at 6 hunger or less, so
                // sprinting through it is a movement the server cannot
                // reproduce. Grim reports it as SprintA.
                let can_sprint = hunger.is_none_or(|hunger| hunger.food > 6);
                if sprint && can_sprint {
                    sprint_events.write(StartSprintEvent {
                        entity,
                        direction: SprintDirection::Forward,
                    });
                } else {
                    walk_events.write(StartWalkEvent {
                        entity,
                        direction: WalkDirection::Forward,
                    });
                }
                if jump {
                    jump_events.write(JumpEvent { entity });
                }
                commands.entity(entity).insert(NavigationStatus::Following {
                    generation: request.generation,
                });
            }
            FollowerDirective::Wait { target, max_turn } => {
                let (yaw, pitch) = steering_direction(
                    **position,
                    look.y_rot(),
                    look.x_rot(),
                    target,
                    0.0,
                    0.0,
                    max_turn,
                    &settings.follower,
                );
                look.update(LookDirection::new(yaw, pitch));
                stop(entity, &mut walk_events);
                commands.entity(entity).insert(NavigationStatus::Following {
                    generation: request.generation,
                });
            }
            FollowerDirective::Arrived if active.reached_goal => {
                stop(entity, &mut walk_events);
                commands.entity(entity).remove::<ActiveNavigation>();
                commands.entity(entity).insert((
                    NavigationStatus::Arrived {
                        generation: request.generation,
                    },
                    NavigationTerminal,
                ));
                continue;
            }
            FollowerDirective::Arrived
            | FollowerDirective::Stuck { .. }
            | FollowerDirective::Unsafe { .. } => {
                if nav_debug() {
                    eprintln!(
                        "nav leg {} ended {:?} at {:.1},{:.1},{:.1} (reached_goal={})",
                        active.legs,
                        directive,
                        position.x,
                        position.y,
                        position.z,
                        active.reached_goal,
                    );
                }
                stop(entity, &mut walk_events);
                // Remember where this went wrong before planning the next leg.
                // `Arrived` on a partial path is not a failure to blame: the
                // leg did its job and the next one carries on from the end of
                // it. Stuck and Unsafe are the two that repeat forever.
                if let (Some(memory), FollowerDirective::Stuck { at } | FollowerDirective::Unsafe { at, .. }) =
                    (avoid.as_deref_mut(), directive)
                {
                    memory.blame(at);
                }
                let avoid = avoid.as_deref().map(AvoidMemory::snapshot).unwrap_or_default();
                if active.legs >= settings.max_plan_legs.max(1) {
                    commands.entity(entity).remove::<ActiveNavigation>();
                    commands.entity(entity).insert((
                        NavigationStatus::Failed {
                            generation: request.generation,
                        },
                        NavigationTerminal,
                    ));
                } else {
                    let task = spawn_plan(
                        *request,
                        BlockPos::from(&**position),
                        world_holder.shared.clone(),
                        settings.clone(),
                        active.legs + 1,
                        active.stalled_legs,
                        avoid,
                    );
                    commands.entity(entity).remove::<ActiveNavigation>();
                    commands.entity(entity).insert((
                        NavigationTask(task),
                        NavigationStatus::Planning {
                            generation: request.generation,
                        },
                    ));
                }
                continue;
            }
        }

        active.dynamic_ticks = active.dynamic_ticks.saturating_add(1);
        if matches!(request.goal, NavigationGoal::Dynamic { .. })
            && settings.dynamic_replan_ticks > 0
            && active.dynamic_ticks >= settings.dynamic_replan_ticks
        {
            stop(entity, &mut walk_events);
            let task = spawn_plan(
                *request,
                BlockPos::from(&**position),
                world_holder.shared.clone(),
                settings.clone(),
                active.legs + 1,
                active.stalled_legs,
                avoid.as_deref().map(AvoidMemory::snapshot).unwrap_or_default(),
            );
            commands.entity(entity).remove::<ActiveNavigation>();
            commands.entity(entity).insert((
                NavigationTask(task),
                NavigationStatus::Planning {
                    generation: request.generation,
                },
            ));
        }
    }
}

fn stop(entity: Entity, events: &mut MessageWriter<StartWalkEvent>) {
    events.write(StartWalkEvent {
        entity,
        direction: WalkDirection::None,
    });
}

#[allow(clippy::type_complexity)]
fn stop_removed_navigation(
    mut commands: Commands,
    query: Query<
        Entity,
        (
            With<LocalEntity>,
            Without<NavigationRequest>,
            Or<(
                With<NavigationTask>,
                With<ActiveNavigation>,
                With<NavigationTerminal>,
            )>,
        ),
    >,
    mut walk_events: MessageWriter<StartWalkEvent>,
) {
    for entity in &query {
        stop(entity, &mut walk_events);
        commands
            .entity(entity)
            .remove::<(NavigationTask, ActiveNavigation, NavigationTerminal)>();
        commands.entity(entity).insert(NavigationStatus::Idle);
    }
}

struct UnavailableWorld;

impl crate::WorldView for UnavailableWorld {
    fn block(&self, _pos: BlockPos) -> crate::BlockKind {
        crate::BlockKind::Unloaded
    }
}

fn follower_snapshot_bounds(
    position: azalea::Vec3,
    path: &crate::Path,
    index: usize,
    settings: &FollowerSettings,
) -> (BlockPos, BlockPos) {
    if path.nodes.is_empty() {
        let center = BlockPos::from(&position);
        return (center, center);
    }
    let index = index.min(path.nodes.len().saturating_sub(1));
    let limit = index
        .saturating_add(settings.max_los_skip)
        .min(path.nodes.len().saturating_sub(1));
    let mut low = BlockPos::from(&position);
    let mut high = low;
    for node in &path.nodes[index..=limit] {
        low = BlockPos::new(
            low.x.min(node.pos.x),
            low.y.min(node.pos.y),
            low.z.min(node.pos.z),
        );
        high = BlockPos::new(
            high.x.max(node.pos.x),
            high.y.max(node.pos.y),
            high.z.max(node.pos.z),
        );
    }
    let clearance = match settings.lava_policy {
        crate::LavaPolicy::Forbidden { clearance } => clearance.max(0),
        crate::LavaPolicy::Penalized => 0,
    };
    let margin = clearance.saturating_add(1);
    (
        BlockPos::new(
            low.x.saturating_sub(margin),
            low.y.saturating_sub(margin),
            low.z.saturating_sub(margin),
        ),
        BlockPos::new(
            high.x.saturating_add(margin),
            high.y.saturating_add(margin),
            high.z.saturating_add(margin),
        ),
    )
}

fn snapshot_bounds(
    start: BlockPos,
    target: BlockPos,
    settings: &PathfinderSettings,
) -> (BlockPos, BlockPos) {
    let axis = |start: i32, target: i32, max_span: i32| {
        let margin = settings.snapshot_margin.max(0);
        let mut low = start.min(target).saturating_sub(margin);
        let mut high = start.max(target).saturating_add(margin);
        if i64::from(high) - i64::from(low) > i64::from(max_span) {
            if target >= start {
                low = start.saturating_sub(margin);
                high = low.saturating_add(max_span);
            } else {
                high = start.saturating_add(margin);
                low = high.saturating_sub(max_span);
            }
        }
        (low, high)
    };
    let (lx, hx) = axis(start.x, target.x, settings.snapshot_max_span_xz.max(1));
    let (ly, hy) = axis(start.y, target.y, settings.snapshot_max_span_y.max(1));
    let (lz, hz) = axis(start.z, target.z, settings.snapshot_max_span_xz.max(1));
    (BlockPos::new(lx, ly, lz), BlockPos::new(hx, hy, hz))
}

pub trait AzaleaPathfinderClientExt {
    fn start_navigation(&self, request: NavigationRequest);
    fn pause_navigation(&self, paused: bool);
    fn cancel_navigation(&self);
    fn navigation_status(&self) -> NavigationStatus;
    fn wait_for_navigation(&self, generation: u64) -> impl Future<Output = NavigationStatus>;
}

impl AzaleaPathfinderClientExt for Client {
    fn start_navigation(&self, request: NavigationRequest) {
        self.ecs.write().entity_mut(self.entity).insert(request);
    }
    fn pause_navigation(&self, paused: bool) {
        self.ecs
            .write()
            .entity_mut(self.entity)
            .insert(NavigationPaused(paused));
        if paused {
            self.walk(WalkDirection::None);
        }
    }
    fn cancel_navigation(&self) {
        self.walk(WalkDirection::None);
        self.ecs
            .write()
            .entity_mut(self.entity)
            .remove::<NavigationRequest>();
    }
    fn navigation_status(&self) -> NavigationStatus {
        self.component::<NavigationStatus>()
            .map(|status| *status)
            .unwrap_or_default()
    }
    async fn wait_for_navigation(&self, generation: u64) -> NavigationStatus {
        let mut ticks = self.get_tick_broadcaster();
        loop {
            let status = self.navigation_status();
            match status {
                NavigationStatus::Arrived {
                    generation: current,
                }
                | NavigationStatus::Failed {
                    generation: current,
                } if current == generation => return status,
                _ => {}
            }
            let request_generation = self
                .component::<NavigationRequest>()
                .ok()
                .map(|request| request.generation);
            if request_generation.is_some_and(|current| current != generation) {
                return NavigationStatus::Failed { generation };
            }
            if request_generation.is_none() && matches!(status, NavigationStatus::Idle) {
                return NavigationStatus::Idle;
            }
            if ticks.recv().await.is_err() {
                return NavigationStatus::Failed { generation };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_bounds_cap_far_goal_in_its_direction() {
        let settings = PathfinderSettings {
            snapshot_margin: 4,
            snapshot_max_span_xz: 32,
            snapshot_max_span_y: 16,
            ..PathfinderSettings::default()
        };
        let start = BlockPos::new(0, 64, 0);
        let goal = BlockPos::new(500, 20, -500);
        let (low, high) = snapshot_bounds(start, goal, &settings);
        assert_eq!((low.x, high.x), (-4, 28));
        assert_eq!((low.y, high.y), (52, 68));
        assert_eq!((low.z, high.z), (-28, 4));
    }

    #[test]
    fn follower_snapshot_bounds_cover_live_lookahead_and_lava_clearance() {
        let path = crate::Path {
            nodes: vec![
                crate::PathNode {
                    pos: BlockPos::new(10, 64, 10),
                    reached_by: crate::MoveKind::Start,
                },
                crate::PathNode {
                    pos: BlockPos::new(13, 65, 8),
                    reached_by: crate::MoveKind::Walk,
                },
            ],
            total_cost: 10,
        };
        let settings = FollowerSettings {
            max_los_skip: 4,
            lava_policy: crate::LavaPolicy::Forbidden { clearance: 2 },
            ..FollowerSettings::default()
        };
        let (low, high) =
            follower_snapshot_bounds(azalea::Vec3::new(9.5, 64.0, 11.5), &path, 0, &settings);
        assert_eq!(low, BlockPos::new(6, 61, 5));
        assert_eq!(high, BlockPos::new(16, 68, 14));
    }
}
