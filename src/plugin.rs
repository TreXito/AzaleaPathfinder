//! Azalea-facing request/state types.
//!
//! Execution systems are added in the next migration phase. Keeping this API
//! independent from HuntingMacro lets any Azalea client publish goals without
//! exposing its combat or application state to the planner.

use azalea::BlockPos;
use azalea::app::{App, Plugin};
use azalea::ecs::prelude::Resource;

#[derive(Debug, Clone, Resource)]
pub struct PathfinderSettings {
    /// Number of game ticks between dynamic-goal rescans/replans.
    pub dynamic_replan_ticks: u32,
}

impl Default for PathfinderSettings {
    fn default() -> Self {
        Self {
            dynamic_replan_ticks: 5,
        }
    }
}

/// A caller-owned destination. Updating `revision` tells the future executor
/// that a dynamic target moved or changed identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationGoal {
    Fixed(BlockPos),
    Dynamic { position: BlockPos, revision: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NavigationRequest {
    pub goal: NavigationGoal,
    pub path_seed: u64,
    /// Monotonic caller token used to discard superseded work.
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

/// Plugin scaffold for Azalea/Bevy applications. Phase one registers shared
/// settings; request execution will move here after HuntingMacro adopts the
/// request/status boundary.
pub struct AzaleaPathfinderPlugin;

impl Plugin for AzaleaPathfinderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PathfinderSettings>();
    }
}
