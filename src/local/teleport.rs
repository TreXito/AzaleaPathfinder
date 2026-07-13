//! Item-teleport movement rules: extension points, not yet implemented.
//!
//! These exist so the framework's shape is proven: they implement the same
//! `Move` trait as walking, so wiring in real teleport planning never
//! touches the A* core. A real implementation needs:
//!
//! - a line-of-sight raycast through the `WorldView` (teleports require an
//!   unobstructed path to the target block)
//! - range limits (AOTV ~12 blocks, etherwarp ~57 with the merged AOTE)
//! - mana availability, tracked via a field added to `MoveContext`
//! - executor support: these edges can't be walked, so the path follower
//!   must right-click (and sneak, for etherwarp) when it hits one —
//!   dispatch on `PathNode::reached_by` in `executor::walk_to`.

use azalea::BlockPos;

use super::moves::{Edge, Move, MoveContext};
use super::world::WorldView;

/// Aspect of the Void short-range teleport.
pub struct AotvMove;

impl Move for AotvMove {
    fn candidates(
        &self,
        _from: BlockPos,
        _world: &dyn WorldView,
        _ctx: &MoveContext,
        _out: &mut Vec<Edge>,
    ) {
        // inert until implemented — produces no candidate edges
    }
}

/// Etherwarp: sneak + right-click teleport onto a distant block surface.
pub struct EtherwarpMove;

impl Move for EtherwarpMove {
    fn candidates(
        &self,
        _from: BlockPos,
        _world: &dyn WorldView,
        _ctx: &MoveContext,
        _out: &mut Vec<Edge>,
    ) {
        // inert until implemented — produces no candidate edges
    }
}
