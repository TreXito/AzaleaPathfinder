use std::fmt;

use azalea::BlockPos;

/// Path cost in tenths of a block. Walking one block costs 10.
pub type Cost = u32;

/// Movement used to reach a path node.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    Start,
    Walk,
    Jump,
    Fall,
    Aotv,
    Etherwarp,
}

#[derive(Debug, Clone, Copy)]
pub struct PathNode {
    pub pos: BlockPos,
    /// Movement used to reach this node.
    #[allow(dead_code)]
    pub reached_by: MoveKind,
}

/// A block-level path produced by the local A* planner.
#[derive(Debug, Clone)]
pub struct Path {
    pub nodes: Vec<PathNode>,
    pub total_cost: Cost,
}

#[derive(Debug)]
pub enum PathError {
    /// No route reaches the goal.
    NoPath,
    /// The search hit its expansion limit.
    SearchBudgetExhausted,
    /// The follower stopped making progress.
    Stuck { at: BlockPos },
    /// The request was replaced or cancelled.
    Cancelled,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathError::NoPath => write!(f, "no path to the goal"),
            PathError::SearchBudgetExhausted => {
                write!(f, "search budget exhausted before reaching the goal")
            }
            PathError::Stuck { at } => write!(f, "stuck while following path at {at:?}"),
            PathError::Cancelled => write!(f, "navigation was cancelled"),
        }
    }
}

impl std::error::Error for PathError {}
