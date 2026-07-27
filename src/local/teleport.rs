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

    fn metadata(
        &self,
        from: BlockPos,
        edge: &Edge,
        world: &dyn WorldView,
        ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        Some(crate::planning::metadata_from_generated_edge(
            crate::PrimitiveId::AOTV,
            from,
            edge.to,
            edge.kind,
            world,
            ctx,
            edge.cost,
        ))
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

    fn metadata(
        &self,
        from: BlockPos,
        edge: &Edge,
        world: &dyn WorldView,
        ctx: &MoveContext,
    ) -> Option<crate::planning::MotionMetadata> {
        Some(crate::planning::metadata_from_generated_edge(
            crate::PrimitiveId::ETHERWARP,
            from,
            edge.to,
            edge.kind,
            world,
            ctx,
            edge.cost,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockKind, MoveKind, PrimitiveId};

    struct EmptyWorld;

    impl WorldView for EmptyWorld {
        fn block(&self, _pos: BlockPos) -> BlockKind {
            BlockKind::Air
        }
    }

    #[test]
    fn extension_metadata_preserves_authoritative_edge_cost() {
        let from = BlockPos::new(0, 64, 0);
        let context = MoveContext::default();
        for (movement, edge, primitive) in [
            (
                &AotvMove as &dyn Move,
                Edge {
                    to: BlockPos::new(8, 70, 0),
                    kind: MoveKind::Aotv,
                    cost: 123,
                },
                PrimitiveId::AOTV,
            ),
            (
                &EtherwarpMove as &dyn Move,
                Edge {
                    to: BlockPos::new(24, 80, -4),
                    kind: MoveKind::Etherwarp,
                    cost: 456,
                },
                PrimitiveId::ETHERWARP,
            ),
        ] {
            let metadata = movement
                .metadata(from, &edge, &EmptyWorld, &context)
                .expect("teleport edge omitted metadata");
            assert_eq!(metadata.primitive, primitive);
            assert_eq!(metadata.predicted.total(), edge.cost);
            assert_eq!(metadata.predicted.time, edge.cost);
        }
    }
}
