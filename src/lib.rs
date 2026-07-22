//! Path planning and navigation for Azalea clients.

pub mod graph;
pub mod local;
pub mod plugin;
pub mod router;
pub mod types;

pub use graph::{GraphEdge, Place, TravelEdge, WorldGraph};
pub use local::astar::{find_path, find_path_best_effort};
pub use local::follower::{
    FollowerDirective, FollowerFrame, FollowerSettings, PathFollower, furthest_visible,
    line_walkable, node_center, steering_direction,
};
pub use local::moves::{
    ClimbMove, LavaPolicy, Move, MoveContext, MovementCosts, WaterPolicy, default_moves,
    lava_risk, lava_transition_allowed,
};
pub use local::world::{BlockKind, WorldSnapshot, WorldView, classify_state};
pub use plugin::{
    AzaleaPathfinderClientExt, AzaleaPathfinderPlugin, NavigationGoal, NavigationPaused,
    NavigationRequest, NavigationStatus, NavigationSystems, PathfinderSettings,
};
pub use router::{TravelStep, route};
pub use types::{Cost, MoveKind, Path, PathError, PathNode};
