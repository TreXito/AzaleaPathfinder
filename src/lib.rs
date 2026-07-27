//! Path planning and navigation for Azalea clients.

#![forbid(unsafe_code)]

pub mod adaptive;
pub mod adaptive_follower;
mod debug;
pub mod graph;
pub mod local;
pub mod planning;
pub mod plugin;
pub mod resilience;
pub mod router;
pub mod types;

pub use adaptive::{
    AdaptiveMode, AdaptiveProfile, AdaptiveRegime, AdaptiveSettings, AttributionEvidence,
    CostComponents, CostModelSnapshot, GuardSignal, InterruptionReason, JourneyMetrics,
    MoveAttempt, MoveFeatures, MoveObservation, ObservationContext, ObservationError,
    ObservationId, ObservationOutcome, PrimitiveId, ProfileError, ProfileKey, PromotionError,
    PromotionReport, TelemetryQueue, TelemetryWorker, follower_settings_hash, load_profile,
    move_context_hash, movement_costs_hash, next_observation_nonce, profile_path,
    save_profile_atomic,
};
pub use adaptive_follower::{ObservedPathFollower, observed_follower_settings};
pub use graph::{GraphEdge, Place, TravelEdge, WorldGraph};
pub use local::astar::{
    BlockGoal, Goal, RevisionKind, RevisionMismatch, SearchAssumptions, SearchSession, SearchSlice,
    SearchStatus, find_motion_plan_best_effort, find_path, find_path_best_effort,
    find_path_to_goal, find_path_to_goal_best_effort,
};
pub use local::follower::{
    FollowerDirective, FollowerFailure, FollowerFrame, FollowerProgress, FollowerSettings,
    PathFollower, furthest_visible, line_walkable, node_center, steering_direction,
};
pub use local::moves::{
    ClimbMove, LavaPolicy, MAX_LAVA_SCAN_RADIUS, Move, MoveContext, MovementCosts, WaterPolicy,
    default_moves, lava_risk, lava_transition_allowed,
};
pub use local::world::{
    BlockKind, MAX_SNAPSHOT_BLOCKS, SnapshotError, WorldSnapshot, WorldView, classify_state,
};
pub use planning::{MotionMetadata, MotionPlan, MotionPlanError, MotionStep};
pub use plugin::{
    AdaptivePathfinderRuntime, AdaptivePathfinderSettings, AzaleaPathfinderClientExt,
    AzaleaPathfinderPlugin, NavigationGoal, NavigationPaused, NavigationRequest, NavigationStatus,
    NavigationSystems, PathfinderSettings, ProfileInstallPolicy,
};
pub use router::{TravelStep, route};
pub use types::{Cost, MoveKind, Path, PathError, PathNode};
