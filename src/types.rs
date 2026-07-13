use std::fmt;

use azalea::BlockPos;

/// Integer path cost in tenths of a block (walking one block costs 10).
/// Integers keep A* ordering exact and `Ord`-friendly.
pub type Cost = u32;

/// How a node in a path is reached from its predecessor.
///
/// `Aotv`/`Etherwarp` pair with the inert teleport moves (see
/// `local::teleport`); they're produced once those moves are implemented and
/// are asserted on in tests, so allow(dead_code) until the executor dispatches
/// on them.
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
    /// How this node was reached. Read by tests today and by the executor once
    /// it dispatches teleport moves (`local::teleport`); see that module's docs.
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
    /// The search exhausted every reachable node without touching the goal.
    NoPath,
    /// The search hit its expansion budget before reaching the goal.
    SearchBudgetExhausted,
    /// The bot stopped making progress while following a path.
    Stuck { at: BlockPos },
    /// A newer request or caller cancellation superseded this path.
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
