use std::collections::HashMap;

use azalea::BlockPos;
use azalea::block::properties::{SlabKind, TopBottom};
use azalea::block::{BlockState, Property};
use azalea::registry::builtin::BlockKind as AzaleaBlockKind;
use azalea::world::World;

/// Coarse classification of a block for pathfinding purposes.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Air,
    Solid,
    /// A fence, wall, or gate. It blocks movement and cannot be used as a floor.
    Fence,
    /// A bottom slab, bottom stair, or carpet that can be walked onto.
    Step,
    /// Lava. It remains occupiable so a trapped bot can plan an exit.
    Lava,
    /// Missing chunk data, treated as impassable.
    Unloaded,
}

/// The block data needed by the local planner.
pub trait WorldView {
    fn block(&self, pos: BlockPos) -> BlockKind;

    /// Returns whether the bot can stand with its feet at `pos`.
    /// Lava is occupiable here; the movement policy decides whether entry is safe.
    fn standable(&self, pos: BlockPos) -> bool {
        let occupiable = |k: BlockKind| matches!(k, BlockKind::Air | BlockKind::Lava);
        // Steps are separate because they raise the body half a block.
        let supportive = |k: BlockKind| matches!(k, BlockKind::Solid | BlockKind::Lava);
        match self.block(pos) {
            BlockKind::Air | BlockKind::Lava => {
                occupiable(self.block(offset(pos, 0, 1, 0)))
                    && supportive(self.block(offset(pos, 0, -1, 0)))
            }
            BlockKind::Step => {
                self.block(offset(pos, 0, 1, 0)) == BlockKind::Air
                    && self.block(offset(pos, 0, 2, 0)) == BlockKind::Air
            }
            _ => false,
        }
    }
}

pub fn offset(p: BlockPos, dx: i32, dy: i32, dz: i32) -> BlockPos {
    BlockPos::new(p.x + dx, p.y + dy, p.z + dz)
}

/// Reduces a Minecraft block state to the geometry used by the planner.
pub fn classify_state(state: BlockState) -> BlockKind {
    if state.is_air() {
        return BlockKind::Air;
    }
    let name = format!("{:?}", AzaleaBlockKind::from(state));
    if name == "Lava" {
        return BlockKind::Lava;
    }
    if name.ends_with("Slab") {
        return match SlabKind::try_from_block_state(state) {
            Some(SlabKind::Bottom) => BlockKind::Step,
            _ => BlockKind::Solid,
        };
    }
    if name.ends_with("Stairs") {
        return match TopBottom::try_from_block_state(state) {
            Some(TopBottom::Bottom) => BlockKind::Step,
            _ => BlockKind::Solid,
        };
    }
    if name.ends_with("Carpet") {
        return BlockKind::Step;
    }
    // These are taller than a full block and cannot be climbed from the side.
    if name.ends_with("Fence") || name.ends_with("FenceGate") || name.ends_with("Wall") {
        return BlockKind::Fence;
    }
    // Swimming and thin-block geometry are not modeled yet.
    BlockKind::Solid
}

impl WorldView for World {
    fn block(&self, pos: BlockPos) -> BlockKind {
        match self.chunks.get_block_state(pos) {
            None => BlockKind::Unloaded,
            Some(state) => classify_state(state),
        }
    }
}

/// An owned copy of a bounded world region.
///
/// Planning against a snapshot avoids holding the world lock during A*. Cells
/// outside the captured area are [`BlockKind::Unloaded`].
pub struct WorldSnapshot {
    /// Dense X/Z/Y storage over `lo..=hi`.
    blocks: Vec<BlockKind>,
    lo: BlockPos,
    hi: BlockPos,
    size_y: usize,
    size_z: usize,
}

impl WorldSnapshot {
    /// Copies the inclusive area `lo..=hi` under one world read lock.
    pub fn capture(world: &parking_lot::RwLock<World>, lo: BlockPos, hi: BlockPos) -> Self {
        let inclusive_len = |low: i32, high: i32| -> usize {
            (i64::from(high) - i64::from(low) + 1).max(0) as usize
        };
        let size_x = inclusive_len(lo.x, hi.x);
        let size_y = inclusive_len(lo.y, hi.y);
        let size_z = inclusive_len(lo.z, hi.z);
        let mut blocks = Vec::with_capacity(size_x.saturating_mul(size_y).saturating_mul(size_z));
        let mut memo: HashMap<u32, BlockKind> = HashMap::new();
        let guard = world.read();
        for x in lo.x..=hi.x {
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    let pos = BlockPos::new(x, y, z);
                    let kind = match guard.chunks.get_block_state(pos) {
                        Some(state) => {
                            let id = u32::from(state);
                            *memo.entry(id).or_insert_with(|| classify_state(state))
                        }
                        None => BlockKind::Unloaded,
                    };
                    blocks.push(kind);
                }
            }
        }
        drop(guard);
        Self {
            blocks,
            lo,
            hi,
            size_y,
            size_z,
        }
    }
}

impl WorldView for WorldSnapshot {
    fn block(&self, pos: BlockPos) -> BlockKind {
        if pos.x < self.lo.x
            || pos.x > self.hi.x
            || pos.y < self.lo.y
            || pos.y > self.hi.y
            || pos.z < self.lo.z
            || pos.z > self.hi.z
        {
            return BlockKind::Unloaded;
        }
        let x = (pos.x - self.lo.x) as usize;
        let y = (pos.y - self.lo.y) as usize;
        let z = (pos.z - self.lo.z) as usize;
        self.blocks[(x * self.size_z + z) * self.size_y + y]
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;

    #[test]
    fn dense_snapshot_preserves_air_solid_and_unloaded_space() {
        let snapshot = WorldSnapshot {
            blocks: vec![BlockKind::Unloaded, BlockKind::Solid, BlockKind::Air],
            lo: BlockPos::new(0, 63, 0),
            hi: BlockPos::new(0, 65, 0),
            size_y: 3,
            size_z: 1,
        };

        assert_eq!(snapshot.block(BlockPos::new(0, 63, 0)), BlockKind::Unloaded);
        assert_eq!(snapshot.block(BlockPos::new(0, 64, 0)), BlockKind::Solid);
        assert_eq!(snapshot.block(BlockPos::new(0, 65, 0)), BlockKind::Air);
        assert_eq!(snapshot.block(BlockPos::new(1, 64, 0)), BlockKind::Unloaded);
        assert_eq!(snapshot.block(BlockPos::new(0, 62, 0)), BlockKind::Unloaded);
    }
}
