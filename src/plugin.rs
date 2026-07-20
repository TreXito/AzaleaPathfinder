//! Azalea plugin and client-facing navigation API.

use std::future::Future;

use azalea::StartSprintEvent;
use azalea::StartWalkEvent;
use azalea::app::{App, Plugin};
use azalea::bot::JumpEvent;
use azalea::ecs::prelude::*;
use azalea::entity::{LocalEntity, LookDirection, Physics, Position};
use azalea::local_player::WorldHolder;
use azalea::physics::PhysicsSystems;
use azalea::prelude::GameTick;
use azalea::{BlockPos, Client, SprintDirection, WalkDirection};
use bevy_tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use crate::{
    FollowerDirective, FollowerFrame, FollowerSettings, MoveContext, PathFollower, WorldSnapshot,
    default_moves, find_path_best_effort, steering_direction,
};

#[derive(Debug, Clone, Resource)]
pub struct PathfinderSettings {
    /// Periodic refresh for dynamic goals, even if the publisher did not bump
    /// its revision. A value of zero disables periodic refresh.
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
            dynamic_replan_ticks: 5,
            max_plan_legs: 16,
            snapshot_margin: 10,
            snapshot_max_span_xz: 128,
            snapshot_max_span_y: 64,
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

/// Insert this component on a local-player entity to start or replace a route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct NavigationRequest {
    pub goal: NavigationGoal,
    pub path_seed: u64,
    /// Monotonic caller token. Results from older generations are discarded.
    pub generation: u64,
}

/// Application-owned pause switch. Paused time never consumes the stall budget.
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
    snapshot: WorldSnapshot,
    reached_goal: bool,
    legs: u32,
    dynamic_ticks: u32,
}

#[derive(Component)]
struct NavigationTerminal;

struct PlanResult {
    request: NavigationRequest,
    path: crate::Path,
    snapshot: WorldSnapshot,
    reached_goal: bool,
    legs: u32,
}

/// Complete block-navigation plugin: it plans off-thread, follows paths on the
/// game tick, replans partial/dynamic routes, and publishes status components.
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
                    .chain(),
            )
            .add_systems(GameTick, tick_navigation.after(PhysicsSystems));
    }
}

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
        let task = spawn_plan(
            *request,
            BlockPos::from(&**position),
            world.shared.clone(),
            settings.clone(),
            1,
        );
        commands.entity(entity).insert((
            NavigationTask(task),
            NavigationStatus::Planning {
                generation: request.generation,
            },
        ));
    }
}

fn spawn_plan(
    request: NavigationRequest,
    start: BlockPos,
    world: std::sync::Arc<parking_lot::RwLock<azalea::world::World>>,
    settings: PathfinderSettings,
    legs: u32,
) -> Task<PlanResult> {
    AsyncComputeTaskPool::get().spawn(async move {
        let target = request.goal.position();
        let (lo, hi) = snapshot_bounds(start, target, &settings);
        let snapshot = WorldSnapshot::capture(&world, lo, hi);
        let mut context = settings.planner.clone();
        context.path_seed = request.path_seed;
        let (path, reached_goal) =
            find_path_best_effort(&snapshot, start, target, &default_moves(), &context);
        PlanResult {
            request,
            path,
            snapshot,
            reached_goal,
            legs,
        }
    })
}

fn poll_navigation_tasks(
    mut commands: Commands,
    settings: Res<PathfinderSettings>,
    mut query: Query<(Entity, &NavigationRequest, &mut NavigationTask)>,
) {
    for (entity, current, mut task) in &mut query {
        let Some(result) = future::block_on(future::poll_once(&mut task.0)) else {
            continue;
        };
        commands.entity(entity).remove::<NavigationTask>();
        if *current != result.request {
            continue;
        }
        if result.path.nodes.len() < 2 {
            let status = if result.reached_goal {
                NavigationStatus::Arrived {
                    generation: current.generation,
                }
            } else {
                NavigationStatus::Failed {
                    generation: current.generation,
                }
            };
            commands.entity(entity).insert((status, NavigationTerminal));
            continue;
        }
        let follower = PathFollower::new(result.path, settings.follower.clone(), current.path_seed);
        commands.entity(entity).insert((
            ActiveNavigation {
                follower,
                snapshot: result.snapshot,
                reached_goal: result.reached_goal,
                legs: result.legs,
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
        &LookDirection,
        &WorldHolder,
        &mut ActiveNavigation,
    )>,
    mut walk_events: MessageWriter<StartWalkEvent>,
    mut sprint_events: MessageWriter<StartSprintEvent>,
    mut jump_events: MessageWriter<JumpEvent>,
) {
    for (entity, request, paused, position, physics, world, world_holder, mut active) in &mut query
    {
        let paused = paused.is_some_and(|paused| paused.0);
        let active_mut = &mut *active;
        let directive = active_mut.follower.tick(
            &active_mut.snapshot,
            FollowerFrame {
                position: **position,
                on_ground: physics.on_ground(),
                horizontal_collision: physics.horizontal_collision,
                paused,
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
                    world.y_rot(),
                    world.x_rot(),
                    target,
                    yaw_bias,
                    pitch_bias,
                    max_turn,
                    &settings.follower,
                );
                commands
                    .entity(entity)
                    .insert(LookDirection::new(yaw, pitch));
                if sprint {
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
                stop(entity, &mut walk_events);
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
            let task = spawn_plan(
                *request,
                BlockPos::from(&**position),
                world_holder.shared.clone(),
                settings.clone(),
                active.legs + 1,
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

fn snapshot_bounds(
    start: BlockPos,
    target: BlockPos,
    settings: &PathfinderSettings,
) -> (BlockPos, BlockPos) {
    let axis = |start: i32, target: i32, max_span: i32| {
        let margin = settings.snapshot_margin.max(0);
        let mut low = start.min(target) - margin;
        let mut high = start.max(target) + margin;
        if high - low > max_span {
            if target >= start {
                low = start - margin;
                high = low + max_span;
            } else {
                high = start + margin;
                low = high - max_span;
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
    fn snapshot_bounds_cover_near_goal_with_margin() {
        let settings = PathfinderSettings::default();
        let start = BlockPos::new(10, 64, -20);
        let goal = BlockPos::new(30, 70, -5);
        let (low, high) = snapshot_bounds(start, goal, &settings);
        assert_eq!(low, BlockPos::new(0, 54, -30));
        assert_eq!(high, BlockPos::new(40, 80, 5));
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
    fn dynamic_goal_position_ignores_revision() {
        let position = BlockPos::new(1, 2, 3);
        assert_eq!(
            NavigationGoal::Dynamic {
                position,
                revision: 99,
            }
            .position(),
            position
        );
    }
}
