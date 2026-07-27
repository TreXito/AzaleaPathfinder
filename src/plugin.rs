//! Azalea plugin and navigation API.

use std::future::Future;

use azalea::StartSprintEvent;
use azalea::StartWalkEvent;
use azalea::app::{App, Plugin};
use azalea::bot::JumpEvent;
use azalea::ecs::prelude::*;
use azalea::entity::metadata::Health;
use azalea::entity::{LocalEntity, LookDirection, Physics, Position};
use azalea::local_player::{Hunger, WorldHolder};
use azalea::physics::PhysicsSystems;
use azalea::prelude::GameTick;
use azalea::{BlockPos, Client, SprintDirection, WalkDirection};
use bevy_tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use crate::adaptive::{ProfileError, load_profile_unprepared};
use crate::{
    AdaptiveMode, AdaptiveProfile, AdaptiveRegime, AdaptiveSettings, CostModelSnapshot,
    FollowerDirective, FollowerFrame, FollowerSettings, GuardSignal, InterruptionReason,
    MotionPlan, MoveContext, MoveObservation, MovementCosts, ObservationContext,
    ObservedPathFollower, PathFollower, ProfileKey, PromotionError, PromotionReport, WorldSnapshot,
    WorldView, default_moves, find_motion_plan_best_effort, follower_settings_hash,
    move_context_hash, movement_costs_hash, next_observation_nonce, observed_follower_settings,
    steering_direction,
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

/// Opt-in adaptive routing configuration.
///
/// The default is shadow-only and has no profile key, so installing the plugin
/// cannot silently persist data or alter route selection. Set an explicit
/// [`ProfileKey`] before enabling a promoted model.
#[derive(Debug, Clone, Resource)]
pub struct AdaptivePathfinderSettings {
    pub profile: Option<ProfileKey>,
    pub learner: AdaptiveSettings,
    pub build_id: String,
    pub actor_capability_hash: u64,
    /// Application-maintained revision for the approximately captured world
    /// view. Live follower validation remains authoritative.
    pub world_revision: u64,
}

impl Default for AdaptivePathfinderSettings {
    fn default() -> Self {
        Self {
            profile: None,
            learner: AdaptiveSettings::default(),
            build_id: env!("CARGO_PKG_VERSION").into(),
            actor_capability_hash: 0,
            world_revision: 1,
        }
    }
}

/// In-memory bounded learner used by the plugin. Disk persistence and JSONL
/// telemetry remain explicit maintenance-thread operations.
#[derive(Resource)]
pub struct AdaptivePathfinderRuntime {
    profiles: parking_lot::Mutex<std::collections::BTreeMap<ProfileKey, AdaptiveProfile>>,
    dropped_observations: std::sync::atomic::AtomicU64,
}

const MAX_ADAPTIVE_PROFILES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileInstallPolicy {
    RejectExisting,
    ReplaceExisting,
}

impl Default for AdaptivePathfinderRuntime {
    fn default() -> Self {
        Self {
            profiles: parking_lot::Mutex::default(),
            dropped_observations: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl AdaptivePathfinderRuntime {
    pub fn allocate_journey_id(&self) -> u64 {
        next_observation_nonce()
    }

    pub fn snapshot(
        &self,
        key: &ProfileKey,
        baseline: MovementCosts,
        settings: &AdaptiveSettings,
        regime: &AdaptiveRegime,
    ) -> CostModelSnapshot {
        if let Some(mut profiles) = self.profiles.try_lock() {
            if !profiles.contains_key(key) && profiles.len() >= MAX_ADAPTIVE_PROFILES {
                self.dropped_observations
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return AdaptiveProfile::new(key.clone()).snapshot(baseline, settings, regime);
            }
            return profiles
                .entry(key.clone())
                .or_insert_with(|| AdaptiveProfile::new(key.clone()))
                .snapshot(baseline, settings, regime);
        }
        // Contention must never block GameTick or planning dispatch. A fresh
        // profile has no active model, so this is a safe baseline snapshot.
        self.dropped_observations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        AdaptiveProfile::new(key.clone()).snapshot(baseline, settings, regime)
    }

    pub fn try_observe(&self, observation: &MoveObservation, settings: &AdaptiveSettings) -> bool {
        if !observation.validate() {
            self.dropped_observations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        let Some(mut profiles) = self.profiles.try_lock() else {
            self.dropped_observations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        };
        if !profiles.contains_key(&observation.context.profile)
            && profiles.len() >= MAX_ADAPTIVE_PROFILES
        {
            self.dropped_observations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        profiles
            .entry(observation.context.profile.clone())
            .or_insert_with(|| AdaptiveProfile::new(observation.context.profile.clone()))
            .observe(observation, settings)
    }

    pub fn promote(
        &self,
        key: &ProfileKey,
        report: &PromotionReport,
        settings: &AdaptiveSettings,
    ) -> Result<u64, PromotionError> {
        let mut profiles = self.profiles.lock();
        if !profiles.contains_key(key) && profiles.len() >= MAX_ADAPTIVE_PROFILES {
            return Err(PromotionError::ProfileCapacity);
        }
        profiles
            .entry(key.clone())
            .or_insert_with(|| AdaptiveProfile::new(key.clone()))
            .promote(report, settings)
    }

    pub fn rollback(&self, key: &ProfileKey, signal: GuardSignal) -> Result<u64, PromotionError> {
        let mut profiles = self.profiles.lock();
        if !profiles.contains_key(key) && profiles.len() >= MAX_ADAPTIVE_PROFILES {
            return Err(PromotionError::ProfileCapacity);
        }
        profiles
            .entry(key.clone())
            .or_insert_with(|| AdaptiveProfile::new(key.clone()))
            .rollback(signal)
    }

    pub fn profile(&self, key: &ProfileKey) -> Option<AdaptiveProfile> {
        self.profiles.lock().get(key).cloned()
    }

    /// Installs a fully validated profile from a maintenance thread. Callers
    /// must choose explicitly whether an existing in-memory profile may be
    /// replaced.
    pub fn install_profile(
        &self,
        profile: AdaptiveProfile,
        policy: ProfileInstallPolicy,
    ) -> Result<Option<AdaptiveProfile>, ProfileError> {
        if !profile.validate() {
            return Err(ProfileError::Invalid);
        }
        let mut profiles = self.profiles.lock();
        let exists = profiles.contains_key(&profile.key);
        if exists && policy == ProfileInstallPolicy::RejectExisting {
            return Err(ProfileError::AlreadyExists);
        }
        if !exists && profiles.len() >= MAX_ADAPTIVE_PROFILES {
            return Err(ProfileError::Capacity);
        }
        profile.prepare_nonce_allocator()?;
        Ok(profiles.insert(profile.key.clone(), profile))
    }

    /// Loads and installs one persisted profile. This performs blocking I/O
    /// and is therefore intended for startup or another maintenance thread.
    pub fn load_from(
        &self,
        root: &std::path::Path,
        key: &ProfileKey,
        policy: ProfileInstallPolicy,
    ) -> Result<Option<AdaptiveProfile>, ProfileError> {
        self.install_profile(load_profile_unprepared(root, key)?, policy)
    }

    pub fn dropped_observations(&self) -> u64 {
        self.dropped_observations
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Clone)]
struct AdaptiveLegSeed {
    key: ProfileKey,
    model: CostModelSnapshot,
    regime: AdaptiveRegime,
    world_revision: u64,
}

struct PlanLeg {
    journey_id: u64,
    legs: u32,
    stalled_legs: u32,
    avoid: std::sync::Arc<std::collections::HashMap<BlockPos, crate::Cost>>,
    health: f32,
    adaptive: Option<AdaptiveLegSeed>,
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

enum ActiveFollower {
    Plain(Box<PathFollower>),
    Observed(Box<ObservedPathFollower>),
}

impl ActiveFollower {
    fn path(&self) -> &crate::Path {
        match self {
            Self::Plain(follower) => follower.path(),
            Self::Observed(follower) => follower.path(),
        }
    }

    fn current_node_index(&self) -> usize {
        match self {
            Self::Plain(follower) => follower.current_node_index(),
            Self::Observed(follower) => follower.current_node_index(),
        }
    }

    fn tick(&mut self, world: &dyn WorldView, frame: FollowerFrame) -> FollowerDirective {
        match self {
            Self::Plain(follower) => follower.tick(world, frame),
            Self::Observed(follower) => follower.tick(world, frame),
        }
    }

    fn interrupt(&mut self, reason: InterruptionReason) {
        if let Self::Observed(follower) = self {
            follower.interrupt(reason);
        }
    }

    fn disable_observations(&mut self, reason: InterruptionReason) {
        if let Self::Observed(follower) = self {
            follower.disable_observations(reason);
        }
    }

    fn drain_observations(&mut self) -> Vec<MoveObservation> {
        match self {
            Self::Plain(_) => Vec::new(),
            Self::Observed(follower) => follower.drain_observations().collect(),
        }
    }
}

#[derive(Component)]
struct ActiveNavigation {
    follower: ActiveFollower,
    journey_id: u64,
    reached_goal: bool,
    planned_goal: BlockPos,
    legs: u32,
    /// Consecutive legs that ended within [`STALL_RADIUS`] of where they began.
    stalled_legs: u32,
    dynamic_ticks: u32,
    /// Fall limit against which the remaining path was last checked. A
    /// sentinel forces one check when a newly planned route is installed.
    checked_fall_limit: i32,
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
                    let pos = crate::local::world::offset(at, dx, dy, dz);
                    *self.spots.entry(pos).or_insert(0) = self
                        .spots
                        .get(&pos)
                        .copied()
                        .unwrap_or(0)
                        .saturating_add(AVOID_PENALTY);
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
    journey_id: u64,
    plan: Option<MotionPlan>,
    observation_context: Option<ObservationContext>,
    reached_goal: bool,
    target: BlockPos,
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

fn is_transient_free_fall(world: &dyn WorldView, position: BlockPos, on_ground: bool) -> bool {
    if on_ground {
        return false;
    }
    let below = crate::local::world::offset(position, 0, -1, 0);
    ![world.block(position), world.block(below)]
        .into_iter()
        .any(|block| {
            matches!(
                block,
                crate::BlockKind::Water | crate::BlockKind::Lava | crate::BlockKind::Climbable
            )
        })
}

fn next_stalled_retry(
    legs: u32,
    stalled_legs: u32,
    max_plan_legs: u32,
    transient_free_fall: bool,
) -> Option<(u32, u32)> {
    if transient_free_fall {
        return Some((legs, stalled_legs));
    }
    let next_stalled = stalled_legs.saturating_add(1);
    if next_stalled > MAX_STALLED_LEGS || legs >= max_plan_legs.max(1) {
        return None;
    }
    Some((legs.saturating_add(1), next_stalled))
}

/// Plans and follows block paths for local players.
pub struct AzaleaPathfinderPlugin;

impl Plugin for AzaleaPathfinderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PathfinderSettings>()
            .init_resource::<AdaptivePathfinderSettings>()
            .init_resource::<AdaptivePathfinderRuntime>()
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
    adaptive_settings: Res<AdaptivePathfinderSettings>,
    adaptive_runtime: Res<AdaptivePathfinderRuntime>,
    mut query: Query<(
        Entity,
        &NavigationRequest,
        Ref<NavigationRequest>,
        &Position,
        &WorldHolder,
        Option<&Health>,
        Option<&NavigationTask>,
        Option<&mut ActiveNavigation>,
        Option<&NavigationTerminal>,
    )>,
    mut walk_events: MessageWriter<StartWalkEvent>,
) {
    for (entity, request, request_ref, position, world, health, task, mut active, terminal) in
        &mut query
    {
        let changed = request_ref.is_changed();
        if !changed && (task.is_some() || active.is_some() || terminal.is_some()) {
            continue;
        }
        if changed {
            if let Some(active) = active.as_deref_mut() {
                active.follower.interrupt(InterruptionReason::GoalChanged);
                record_observations(
                    &mut active.follower,
                    &adaptive_runtime,
                    &adaptive_settings.learner,
                );
            }
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
            PlanLeg {
                journey_id: adaptive_runtime.allocate_journey_id(),
                legs: 1,
                stalled_legs: 0,
                avoid: std::sync::Arc::default(),
                health: health.map_or(0.0, |health| health.0),
                adaptive: adaptive_leg_seed(&adaptive_settings, &adaptive_runtime, &settings),
            },
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

fn adaptive_leg_seed(
    settings: &AdaptivePathfinderSettings,
    runtime: &AdaptivePathfinderRuntime,
    pathfinder: &PathfinderSettings,
) -> Option<AdaptiveLegSeed> {
    let key = settings.profile.as_ref()?;
    // Enabled mode without an explicit, valid production profile never gets
    // here because `profile` is required. Invalid keys likewise fail closed.
    if settings.learner.mode == AdaptiveMode::Disabled
        || !key.validate()
        || key.capability_hash != settings.actor_capability_hash
    {
        return None;
    }
    let observed_settings = observed_follower_settings(pathfinder.follower.clone());
    let regime = AdaptiveRegime {
        build_id: settings.build_id.clone(),
        planner_settings_hash: move_context_hash(&pathfinder.planner),
        control_settings_hash: follower_settings_hash(&observed_settings),
        baseline_costs_hash: movement_costs_hash(pathfinder.planner.costs),
        actor_capability_hash: settings.actor_capability_hash,
    };
    if !regime.validate_for(key) {
        return None;
    }
    Some(AdaptiveLegSeed {
        key: key.clone(),
        model: runtime.snapshot(key, pathfinder.planner.costs, &settings.learner, &regime),
        regime,
        world_revision: settings.world_revision,
    })
}

fn record_observations(
    follower: &mut ActiveFollower,
    runtime: &AdaptivePathfinderRuntime,
    settings: &AdaptiveSettings,
) {
    for observation in follower.drain_observations() {
        runtime.try_observe(&observation, settings);
    }
}

/// Conservative fall cap from current health, reserving two hearts for
/// latency, rounding and unrelated damage. Armor/effects are deliberately not
/// credited because the planner does not model them.
fn survivable_fall_limit(health: f32) -> i32 {
    const RESERVED_HEALTH_POINTS: f32 = 4.0;
    if !health.is_finite() || health <= RESERVED_HEALTH_POINTS {
        return crate::local::moves::SAFE_FALL;
    }
    let spendable = (health - RESERVED_HEALTH_POINTS).floor() as i32;
    crate::local::moves::SAFE_FALL.saturating_add(spendable.max(0))
}

fn remaining_falls_are_survivable(follower: &ActiveFollower, fall_limit: i32) -> bool {
    let path = follower.path();
    let index = follower.current_node_index();
    path.nodes
        .windows(2)
        .enumerate()
        .skip(index.saturating_sub(1))
        .all(|(_, nodes)| {
            !matches!(
                nodes[1].reached_by,
                crate::MoveKind::Fall | crate::MoveKind::Parkour { .. }
            ) || nodes[0].pos.y.saturating_sub(nodes[1].pos.y) <= fall_limit
        })
}

fn spawn_plan(
    request: NavigationRequest,
    start: BlockPos,
    world: std::sync::Arc<parking_lot::RwLock<azalea::world::World>>,
    settings: PathfinderSettings,
    leg: PlanLeg,
) -> Task<PlanResult> {
    let PlanLeg {
        journey_id,
        legs,
        stalled_legs,
        avoid,
        health,
        adaptive,
    } = leg;
    if crate::debug::navigation_enabled() {
        eprintln!(
            "nav plan leg {legs} from {start:?} with {} tolled blocks",
            avoid.len()
        );
    }
    AsyncComputeTaskPool::get().spawn(async move {
        let started = std::time::Instant::now();
        let total_budget = std::time::Duration::from_millis(settings.planner.time_budget_ms);
        let target = request.goal.position();
        let (lo, hi) = snapshot_bounds(start, target, &settings);
        let Ok(snapshot) = WorldSnapshot::try_capture(&world, lo, hi) else {
            return PlanResult {
                request,
                journey_id,
                plan: None,
                observation_context: None,
                reached_goal: false,
                target,
                legs,
                stalled_legs,
                start,
            };
        };
        let Some(remaining) = total_budget.checked_sub(started.elapsed()) else {
            return PlanResult {
                request,
                journey_id,
                plan: None,
                observation_context: None,
                reached_goal: false,
                target,
                legs,
                stalled_legs,
                start,
            };
        };
        if remaining.is_zero() {
            return PlanResult {
                request,
                journey_id,
                plan: None,
                observation_context: None,
                reached_goal: false,
                target,
                legs,
                stalled_legs,
                start,
            };
        }
        let mut context = settings.planner.clone();
        context.avoid = avoid;
        context.max_fall = context.max_fall.min(survivable_fall_limit(health));
        let baseline_costs = context.costs;
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
        let Some(remaining) = total_budget.checked_sub(started.elapsed()) else {
            return PlanResult {
                request,
                journey_id,
                plan: None,
                observation_context: None,
                reached_goal: false,
                target,
                legs,
                stalled_legs,
                start,
            };
        };
        if remaining.is_zero() {
            return PlanResult {
                request,
                journey_id,
                plan: None,
                observation_context: None,
                reached_goal: false,
                target,
                legs,
                stalled_legs,
                start,
            };
        }
        context.time_budget_ms = remaining.as_millis().clamp(1, u128::from(u64::MAX)) as u64;
        let moves = default_moves();
        let planned = find_motion_plan_best_effort(
            &snapshot,
            start,
            target,
            &moves,
            &context,
            adaptive.as_ref().map(|adaptive| &adaptive.model),
        );
        let Ok((plan, reached_goal)) = planned else {
            return PlanResult {
                request,
                journey_id,
                plan: None,
                observation_context: None,
                reached_goal: false,
                target,
                legs,
                stalled_legs,
                start,
            };
        };
        let observation_context = adaptive.as_ref().map(|adaptive| ObservationContext {
            profile: adaptive.key.clone(),
            journey_id,
            generation: request.generation,
            leg: legs,
            plan_revision: request
                .generation
                .rotate_left(17)
                .wrapping_add(u64::from(legs)),
            world_revision: adaptive.world_revision,
            model_revision: adaptive.model.model_revision,
            planner_settings_hash: adaptive.regime.planner_settings_hash,
            control_settings_hash: adaptive.regime.control_settings_hash,
            baseline_costs_hash: movement_costs_hash(baseline_costs),
            actor_capability_hash: adaptive.regime.actor_capability_hash,
            build_id: adaptive.regime.build_id.clone(),
            created_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| {
                    duration.as_millis().min(u128::from(u64::MAX)) as u64
                }),
        });
        PlanResult {
            request,
            journey_id,
            plan: Some(plan),
            observation_context,
            reached_goal,
            target,
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
fn snap_to_standable(
    world: &WorldSnapshot,
    goal: BlockPos,
    lo: BlockPos,
    hi: BlockPos,
) -> Option<BlockPos> {
    // A goal outside the snapshot is a goal we know nothing about, and snapping
    // it lands somewhere the box happens to end rather than somewhere the world
    // does. Climbing the hub mountain from sea level hit this exactly: the
    // summit sat above the snapshot's Y ceiling, so the goal was relocated 88
    // blocks down onto the highest ledge still inside the box, the search
    // honestly reached it, and the bot reported Arrived while standing
    // three quarters of the way up. Leave a goal we cannot see where it is and
    // let the next leg, planned from higher up, see it properly.
    if goal.y < lo.y
        || goal.y > hi.y
        || goal.x < lo.x
        || goal.x > hi.x
        || goal.z < lo.z
        || goal.z > hi.z
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
                let candidate = crate::local::world::offset(goal, dx, dy, dz);
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
    DIR.get_or_init(|| std::env::var("PF_PATH_DIR").ok())
        .as_deref()
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
    let Some(safe_name) = sanitize_profile_name(&profile.name) else {
        return;
    };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let mut out = format!("{safe_name}\n{} {} {}\n", goal.x, goal.y, goal.z);
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
    let path = std::path::Path::new(dir).join(format!("bot-{safe_name}.path"));
    let _ = std::fs::write(path, out);
}

fn sanitize_profile_name(profile_name: &str) -> Option<String> {
    let safe_name = profile_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .take(64)
        .collect::<String>();
    if safe_name.is_empty() {
        return None;
    }
    Some(safe_name)
}

#[allow(clippy::type_complexity)]
fn poll_navigation_tasks(
    mut commands: Commands,
    settings: Res<PathfinderSettings>,
    adaptive_settings: Res<AdaptivePathfinderSettings>,
    adaptive_runtime: Res<AdaptivePathfinderRuntime>,
    mut query: Query<(
        Entity,
        &NavigationRequest,
        &mut NavigationTask,
        &WorldHolder,
        Option<&AvoidMemory>,
        &Physics,
        Option<&Health>,
        Option<&azalea::player::GameProfileComponent>,
    )>,
) {
    for (entity, current, mut task, world_holder, avoid, physics, health, profile) in &mut query {
        let avoid = avoid.map(AvoidMemory::snapshot).unwrap_or_default();
        let Some(result) = future::block_on(future::poll_once(&mut task.0)) else {
            continue;
        };
        commands.entity(entity).remove::<NavigationTask>();
        if *current != result.request {
            continue;
        }
        let Some(plan) = result.plan else {
            commands.entity(entity).insert((
                NavigationStatus::Failed {
                    generation: current.generation,
                },
                NavigationTerminal,
            ));
            continue;
        };
        let path = plan.path();
        if result.reached_goal && path.nodes.len() < 2 {
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
        let end = path.nodes.last().map_or(result.start, |node| node.pos);
        let travelled = u64::from(end.x.abs_diff(result.start.x))
            + u64::from(end.y.abs_diff(result.start.y))
            + u64::from(end.z.abs_diff(result.start.z));
        if !result.reached_goal && travelled < STALL_RADIUS as u64 {
            // A block position taken mid-fall is not a place the planner can
            // reason about: the feet are in air with air underneath, so no move
            // applies and the plan comes back empty however good the route is.
            // Falling off the mountain's west face burned the entire stall
            // budget in under a second this way and reported Failed while the
            // bot was still in the air. Retrying without spending budget lets
            // the bot land and plan from somewhere real.
            //
            // A ladder or water column is not mid-fall, though. `on_ground` is
            // false throughout both, so treating either as airborne exempts a
            // dead-end climb/swim from the retry budget forever. The exemption
            // is only for a body in free flight, a state that ends by itself
            // quickly; sustained climb/swim states are places the planner can
            // reason about and must consume bounded retries.
            let airborne = {
                let world = world_holder.shared.read();
                is_transient_free_fall(&*world, result.start, physics.on_ground())
            };
            let Some((legs, stalled_legs)) = next_stalled_retry(
                result.legs,
                result.stalled_legs,
                settings.max_plan_legs,
                airborne,
            ) else {
                commands.entity(entity).insert((
                    NavigationStatus::Failed {
                        generation: current.generation,
                    },
                    NavigationTerminal,
                ));
                continue;
            };
            {
                // Straight back to the planner with a fresh seed rather than
                // handing the follower a path to nowhere.
                let task = spawn_plan(
                    *current,
                    result.start,
                    world_holder.shared.clone(),
                    settings.clone(),
                    PlanLeg {
                        journey_id: result.journey_id,
                        legs,
                        stalled_legs,
                        avoid: avoid.clone(),
                        health: health.map_or(0.0, |health| health.0),
                        adaptive: adaptive_leg_seed(
                            &adaptive_settings,
                            &adaptive_runtime,
                            &settings,
                        ),
                    },
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
        if crate::debug::navigation_enabled() {
            let end = path.nodes.last().map(|node| node.pos);
            eprintln!(
                "nav leg {} from {:?} -> {} nodes, ends {:?}, reached_goal={}\n  steps: {:?}",
                result.legs,
                result.start,
                path.nodes.len(),
                end,
                result.reached_goal,
                path.nodes
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
        write_path_viz(profile, current.goal.position(), path);
        let follower = if let Some(context) = result.observation_context {
            ActiveFollower::Observed(Box::new(ObservedPathFollower::new(
                plan,
                settings.follower.clone(),
                current.path_seed,
                context,
            )))
        } else {
            ActiveFollower::Plain(Box::new(PathFollower::new(
                plan.into_path(),
                settings.follower.clone(),
                current.path_seed,
            )))
        };
        commands.entity(entity).insert((
            ActiveNavigation {
                follower,
                journey_id: result.journey_id,
                reached_goal: result.reached_goal,
                planned_goal: result.target,
                legs: result.legs,
                stalled_legs: 0,
                dynamic_ticks: 0,
                checked_fall_limit: i32::MIN,
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
    adaptive_settings: Res<AdaptivePathfinderSettings>,
    adaptive_runtime: Res<AdaptivePathfinderRuntime>,
    mut query: Query<(
        Entity,
        &NavigationRequest,
        Option<&NavigationPaused>,
        &Position,
        &Physics,
        &mut LookDirection,
        &WorldHolder,
        Option<&Hunger>,
        Option<&Health>,
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
        health,
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
            record_observations(
                &mut active.follower,
                &adaptive_runtime,
                &adaptive_settings.learner,
            );
            stop(entity, &mut walk_events);
            commands.entity(entity).insert(NavigationStatus::Paused {
                generation: request.generation,
            });
            continue;
        }

        let live_health = health.map_or(0.0, |health| health.0);
        let live_fall_limit = survivable_fall_limit(live_health);
        let fall_limit_changed = active.checked_fall_limit != live_fall_limit;
        let remaining_falls_safe = !fall_limit_changed
            || remaining_falls_are_survivable(&active.follower, live_fall_limit);
        if fall_limit_changed && remaining_falls_safe {
            active.checked_fall_limit = live_fall_limit;
        }
        if physics.on_ground() && !remaining_falls_safe {
            active.follower.interrupt(InterruptionReason::WorldChanged);
            record_observations(
                &mut active.follower,
                &adaptive_runtime,
                &adaptive_settings.learner,
            );
            stop(entity, &mut walk_events);
            let task = spawn_plan(
                *request,
                BlockPos::from(&**position),
                world_holder.shared.clone(),
                settings.clone(),
                PlanLeg {
                    journey_id: active.journey_id,
                    legs: active.legs,
                    stalled_legs: active.stalled_legs,
                    avoid: avoid
                        .as_deref()
                        .map(AvoidMemory::snapshot)
                        .unwrap_or_default(),
                    health: live_health,
                    adaptive: adaptive_leg_seed(&adaptive_settings, &adaptive_runtime, &settings),
                },
            );
            commands.entity(entity).remove::<ActiveNavigation>();
            commands.entity(entity).insert((
                NavigationTask(task),
                NavigationStatus::Planning {
                    generation: request.generation,
                },
            ));
            continue;
        }

        // Low hunger changes the executor from requested sprinting to walking,
        // which is a different timing regime. Keep following safely, but do
        // not let that partial leg contaminate sprint-capable calibration.
        let can_sprint = hunger.is_none_or(|hunger| hunger.food > 6);
        if !can_sprint {
            active
                .follower
                .disable_observations(InterruptionReason::Unknown);
        }

        // Check nearby blocks again before following the planned path.
        let (low, high) = follower_snapshot_bounds(
            **position,
            active.follower.path(),
            active.follower.current_node_index(),
            &settings.follower,
        );
        let Ok(live_snapshot) = WorldSnapshot::try_capture(&world_holder.shared, low, high) else {
            active.follower.interrupt(InterruptionReason::WorldChanged);
            record_observations(
                &mut active.follower,
                &adaptive_runtime,
                &adaptive_settings.learner,
            );
            stop(entity, &mut walk_events);
            commands.entity(entity).remove::<ActiveNavigation>();
            commands.entity(entity).insert((
                NavigationStatus::Failed {
                    generation: request.generation,
                },
                NavigationTerminal,
            ));
            continue;
        };
        let directive = active.follower.tick(
            &live_snapshot,
            FollowerFrame {
                position: **position,
                on_ground: physics.on_ground(),
                horizontal_collision: physics.horizontal_collision,
                paused: false,
            },
        );
        record_observations(
            &mut active.follower,
            &adaptive_runtime,
            &adaptive_settings.learner,
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
            FollowerDirective::Arrived
                if active.reached_goal
                    && position_reaches_goal(
                        **position,
                        active.planned_goal,
                        settings.planner.goal_tolerance,
                    ) =>
            {
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
                if crate::debug::navigation_enabled() {
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
                if let (
                    Some(memory),
                    FollowerDirective::Stuck { at } | FollowerDirective::Unsafe { at, .. },
                ) = (avoid.as_deref_mut(), directive)
                {
                    memory.blame(at);
                }
                let avoid = avoid
                    .as_deref()
                    .map(AvoidMemory::snapshot)
                    .unwrap_or_default();
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
                        PlanLeg {
                            journey_id: active.journey_id,
                            legs: active.legs.saturating_add(1),
                            stalled_legs: active.stalled_legs,
                            avoid,
                            health: health.map_or(0.0, |health| health.0),
                            adaptive: adaptive_leg_seed(
                                &adaptive_settings,
                                &adaptive_runtime,
                                &settings,
                            ),
                        },
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
            active.follower.interrupt(InterruptionReason::GoalChanged);
            record_observations(
                &mut active.follower,
                &adaptive_runtime,
                &adaptive_settings.learner,
            );
            let task = spawn_plan(
                *request,
                BlockPos::from(&**position),
                world_holder.shared.clone(),
                settings.clone(),
                PlanLeg {
                    journey_id: active.journey_id,
                    legs: active.legs,
                    stalled_legs: active.stalled_legs,
                    avoid: avoid
                        .as_deref()
                        .map(AvoidMemory::snapshot)
                        .unwrap_or_default(),
                    health: health.map_or(0.0, |health| health.0),
                    adaptive: adaptive_leg_seed(&adaptive_settings, &adaptive_runtime, &settings),
                },
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
    adaptive_settings: Res<AdaptivePathfinderSettings>,
    adaptive_runtime: Res<AdaptivePathfinderRuntime>,
    mut query: Query<
        (Entity, Option<&mut ActiveNavigation>),
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
    for (entity, active) in &mut query {
        if let Some(mut active) = active {
            active.follower.interrupt(InterruptionReason::Cancelled);
            record_observations(
                &mut active.follower,
                &adaptive_runtime,
                &adaptive_settings.learner,
            );
        }
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
        crate::LavaPolicy::Forbidden { clearance } => {
            clearance.clamp(0, crate::local::moves::MAX_LAVA_SCAN_RADIUS)
        }
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

fn position_reaches_goal(position: azalea::Vec3, goal: BlockPos, tolerance: i32) -> bool {
    let current = BlockPos::from(&position);
    let distance = u64::from(current.x.abs_diff(goal.x))
        + u64::from(current.y.abs_diff(goal.y))
        + u64::from(current.z.abs_diff(goal.z));
    distance <= u64::from(tolerance.max(0) as u32)
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

    fn adaptive_test_settings() -> AdaptiveSettings {
        AdaptiveSettings {
            mode: AdaptiveMode::Enabled,
            min_samples: 3,
            maximum_failure_upper_ppm: 900_000,
            minimum_evaluation_journeys: 2,
            ..AdaptiveSettings::default()
        }
    }

    fn adaptive_test_regime(key: &ProfileKey, pathfinder: &PathfinderSettings) -> AdaptiveRegime {
        AdaptiveRegime {
            build_id: "plugin-test".into(),
            planner_settings_hash: move_context_hash(&pathfinder.planner),
            control_settings_hash: follower_settings_hash(&observed_follower_settings(
                pathfinder.follower.clone(),
            )),
            baseline_costs_hash: movement_costs_hash(pathfinder.planner.costs),
            actor_capability_hash: key.capability_hash,
        }
    }

    fn adaptive_test_observation(
        key: &ProfileKey,
        regime: &AdaptiveRegime,
        edge_index: u32,
    ) -> MoveObservation {
        crate::MoveAttempt::begin(
            ObservationContext {
                profile: key.clone(),
                journey_id: 1,
                generation: 1,
                leg: 1,
                plan_revision: 1,
                world_revision: 1,
                model_revision: 0,
                planner_settings_hash: regime.planner_settings_hash,
                control_settings_hash: regime.control_settings_hash,
                baseline_costs_hash: regime.baseline_costs_hash,
                actor_capability_hash: regime.actor_capability_hash,
                build_id: regime.build_id.clone(),
                created_unix_ms: 1,
            },
            crate::ObservationId {
                nonce: next_observation_nonce(),
                journey_id: 1,
                generation: 1,
                leg: 1,
                edge_index,
                attempt: 0,
            },
            crate::PrimitiveId::WALK_CARDINAL,
            crate::MoveFeatures::new(1, 0, crate::adaptive::TerrainClass::FullBlock),
            BlockPos::new(edge_index as i32, 64, 0),
            BlockPos::new(edge_index as i32 + 1, 64, 0),
            crate::CostComponents {
                time: 10,
                ..crate::CostComponents::default()
            },
            5,
            0,
        )
        .unwrap()
        .finish(
            10,
            u64::from(edge_index) + 1,
            crate::ObservationOutcome::Success,
            crate::AttributionEvidence::ReachedPlannedNode,
        )
        .unwrap()
    }

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

    #[test]
    fn final_arrival_uses_only_the_planner_tolerance() {
        let goal = BlockPos::new(10, 64, 0);
        assert!(position_reaches_goal(
            azalea::Vec3::new(9.5, 64.0, 0.5),
            goal,
            1
        ));
        assert!(!position_reaches_goal(
            azalea::Vec3::new(7.6, 64.0, 0.5),
            goal,
            1
        ));
    }

    #[test]
    fn health_caps_falls_with_a_two_heart_reserve() {
        assert_eq!(survivable_fall_limit(f32::NAN), 3);
        assert_eq!(survivable_fall_limit(4.0), 3);
        assert_eq!(survivable_fall_limit(5.9), 4);
        assert_eq!(survivable_fall_limit(20.0), 19);
    }

    #[test]
    fn journey_ids_are_unique_across_actors() {
        let first_runtime = AdaptivePathfinderRuntime::default();
        let second_runtime = AdaptivePathfinderRuntime::default();
        let first = first_runtime.allocate_journey_id();
        let second = second_runtime.allocate_journey_id();
        let third = first_runtime.allocate_journey_id();
        assert_ne!(first, 0);
        assert_ne!(first, second);
        assert_ne!(first, third);
        assert_ne!(second, third);
    }

    #[test]
    fn adaptive_seed_rejects_disabled_or_wrong_capability_and_is_health_independent() {
        let runtime = AdaptivePathfinderRuntime::default();
        let pathfinder = PathfinderSettings::default();
        let mut adaptive = AdaptivePathfinderSettings {
            profile: Some(ProfileKey::local_default()),
            learner: adaptive_test_settings(),
            build_id: "plugin-test".into(),
            ..AdaptivePathfinderSettings::default()
        };
        let first = adaptive_leg_seed(&adaptive, &runtime, &pathfinder).unwrap();
        let second = adaptive_leg_seed(&adaptive, &runtime, &pathfinder).unwrap();
        assert_eq!(first.regime, second.regime);
        assert_eq!(
            first.regime.planner_settings_hash,
            move_context_hash(&pathfinder.planner),
            "per-leg health caps must not enter the persistent regime"
        );

        adaptive.actor_capability_hash = 1;
        assert!(adaptive_leg_seed(&adaptive, &runtime, &pathfinder).is_none());
        adaptive.actor_capability_hash = 0;
        adaptive.learner.mode = AdaptiveMode::Disabled;
        assert!(adaptive_leg_seed(&adaptive, &runtime, &pathfinder).is_none());
    }

    #[test]
    fn a_live_health_drop_invalidates_an_upcoming_damaging_fall() {
        let path = crate::Path {
            nodes: vec![
                crate::PathNode {
                    pos: BlockPos::new(0, 64, 0),
                    reached_by: crate::MoveKind::Start,
                },
                crate::PathNode {
                    pos: BlockPos::new(1, 54, 0),
                    reached_by: crate::MoveKind::Fall,
                },
            ],
            total_cost: 10,
        };
        let follower = ActiveFollower::Plain(Box::new(PathFollower::new(
            path,
            FollowerSettings::default(),
            0,
        )));
        assert!(remaining_falls_are_survivable(
            &follower,
            survivable_fall_limit(20.0)
        ));
        assert!(!remaining_falls_are_survivable(
            &follower,
            survivable_fall_limit(10.0)
        ));
    }

    #[test]
    fn swimming_is_not_an_unbounded_free_fall_retry() {
        struct WaterWorld;
        impl WorldView for WaterWorld {
            fn block(&self, _position: BlockPos) -> crate::BlockKind {
                crate::BlockKind::Water
            }
        }
        struct AirWorld;
        impl WorldView for AirWorld {
            fn block(&self, _position: BlockPos) -> crate::BlockKind {
                crate::BlockKind::Air
            }
        }

        let position = BlockPos::new(0, 64, 0);
        assert!(!is_transient_free_fall(&WaterWorld, position, false));
        assert!(is_transient_free_fall(&AirWorld, position, false));

        let mut state = (1, 0);
        let mut retries = 0;
        while let Some(next) = next_stalled_retry(state.0, state.1, 100, false) {
            state = next;
            retries += 1;
            assert!(retries <= MAX_STALLED_LEGS);
        }
        assert_eq!(retries, MAX_STALLED_LEGS);
        assert_eq!(
            next_stalled_retry(1, 0, 100, true),
            Some((1, 0)),
            "only genuine short-lived free fall keeps the retry budget"
        );
    }

    #[test]
    fn invalid_observations_do_not_allocate_profiles_and_capacity_is_bounded() {
        let runtime = AdaptivePathfinderRuntime::default();
        let key = ProfileKey::local_default();
        let pathfinder = PathfinderSettings::default();
        let regime = adaptive_test_regime(&key, &pathfinder);
        let mut invalid = adaptive_test_observation(&key, &regime, 0);
        invalid.schema = 0;
        assert!(!runtime.try_observe(&invalid, &adaptive_test_settings()));
        assert!(runtime.profiles.lock().is_empty());

        for index in 0..MAX_ADAPTIVE_PROFILES {
            let mut key = ProfileKey::local_default();
            key.server = format!("server-{index}");
            runtime
                .install_profile(
                    AdaptiveProfile::new(key),
                    ProfileInstallPolicy::RejectExisting,
                )
                .unwrap();
        }
        let mut overflow = ProfileKey::local_default();
        overflow.server = "overflow".into();
        assert!(matches!(
            runtime.install_profile(
                AdaptiveProfile::new(overflow),
                ProfileInstallPolicy::RejectExisting
            ),
            Err(ProfileError::Capacity)
        ));
    }

    #[test]
    fn persisted_promoted_model_can_be_restored_explicitly() {
        let key = ProfileKey::local_default();
        let pathfinder = PathfinderSettings::default();
        let regime = adaptive_test_regime(&key, &pathfinder);
        let settings = adaptive_test_settings();
        let runtime = AdaptivePathfinderRuntime::default();
        for edge in 0..3 {
            assert!(
                runtime.try_observe(&adaptive_test_observation(&key, &regime, edge), &settings)
            );
        }
        let profile = runtime.profile(&key).unwrap();
        let known_good = crate::JourneyMetrics {
            journeys: 10,
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
            known_good,
            shadow: crate::JourneyMetrics {
                p95_ticks: 99,
                ..known_good
            },
        };
        assert_eq!(runtime.promote(&key, &report, &settings).unwrap(), 1);

        let root = std::env::temp_dir().join(format!(
            "azalea-pathfinder-runtime-profile-{}-{}",
            std::process::id(),
            next_observation_nonce()
        ));
        crate::save_profile_atomic(&root, &runtime.profile(&key).unwrap()).unwrap();
        let restored = AdaptivePathfinderRuntime::default();
        restored
            .load_from(&root, &key, ProfileInstallPolicy::RejectExisting)
            .unwrap();
        let snapshot = restored.snapshot(&key, pathfinder.planner.costs, &settings, &regime);
        assert_eq!(snapshot.model_revision, 1);
        assert!(
            snapshot.edge_cost(
                crate::PrimitiveId::WALK_CARDINAL,
                crate::MoveFeatures::new(1, 0, crate::adaptive::TerrainClass::FullBlock),
                10
            ) > 10
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn path_visualizer_names_cannot_escape_the_configured_directory() {
        assert_eq!(
            sanitize_profile_name("../bad/name\r\n"),
            Some("___bad_name__".to_owned())
        );
        assert_eq!(
            sanitize_profile_name("TreXito_42"),
            Some("TreXito_42".to_owned())
        );
        assert_eq!(sanitize_profile_name(""), None);
    }
}
