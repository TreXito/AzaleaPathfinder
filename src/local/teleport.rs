//! Placeholder item-teleport movement rules.
//!
//! Implementing these requires line-of-sight, range and mana checks, plus
//! follower support for right-clicking and sneaking on teleport edges.

use azalea::BlockPos;

use super::moves::{Edge, Move, MoveContext};
use super::world::WorldView;

/// Placeholder for the Aspect of the Void teleport.
pub struct AotvMove;

impl Move for AotvMove {
    fn candidates(
        &self,
        _from: BlockPos,
        _world: &dyn WorldView,
        _ctx: &MoveContext,
        _out: &mut Vec<Edge>,
    ) {
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }
}

/// Placeholder for the Etherwarp teleport.
pub struct EtherwarpMove;

impl Move for EtherwarpMove {
    fn candidates(
        &self,
        _from: BlockPos,
        _world: &dyn WorldView,
        _ctx: &MoveContext,
        _out: &mut Vec<Edge>,
    ) {
    }

    fn supports_builtin_heuristic(&self) -> bool {
        true
    }
}
