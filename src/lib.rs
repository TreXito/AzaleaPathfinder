//! Reusable path planning for Azalea clients.
//!
//! The first extraction phase intentionally contains no hunting, combat, farm,
//! or `Player` state. Callers choose a goal; this crate plans how to reach it.

pub mod graph;
pub mod local;
pub mod plugin;
pub mod router;
pub mod types;

pub use graph::{GraphEdge, Place, TravelEdge, WorldGraph};
pub use local::astar::{find_path, find_path_best_effort};
pub use local::moves::{Move, MoveContext, default_moves};
pub use local::world::{BlockKind, WorldSnapshot, WorldView};
pub use plugin::{
    AzaleaPathfinderPlugin, NavigationGoal, NavigationRequest, NavigationStatus, PathfinderSettings,
};
pub use router::{TravelStep, route};
pub use types::{Cost, MoveKind, Path, PathError, PathNode};
