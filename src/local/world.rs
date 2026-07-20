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
    /// A fence, wall, or fence gate: rendered thin but 1.5 blocks TALL, so —
    /// unlike a normal solid — the bot can neither pass through it NOR step/jump
    /// onto it (you can't climb or hop a fence in Minecraft). Not occupiable and
    /// deliberately NOT a valid floor (only `Solid`/`Lava` support a stand in
    /// [`WorldView::standable`]), so the planner routes AROUND a fence post
    /// instead of up-and-over it — the over-the-post route it can't execute,
    /// which had it bump the post, jump, get set back, and oscillate in place.
    Fence,
    /// A partial block a player walks onto without jumping: bottom slab,
    /// bottom-half stairs, carpet. The cell itself is enterable (feet
    /// inside it) and it counts as floor for the cell above.
    Step,
    /// Lava (source or flowing). Kept structurally occupiable so a bot already
    /// in it can plan an exit and callers can explicitly select the penalized
    /// legacy policy. The default planner forbids entering its safety buffer.
    Lava,
    /// Chunk not loaded — treated as impassable so paths never leave the
    /// known world. Long-distance travel is the router's job.
    Unloaded,
}

/// What the local planner needs to know about the world. The live azalea
/// world implements this; tests implement it over a hand-built grid.
pub trait WorldView {
    fn block(&self, pos: BlockPos) -> BlockKind;

    /// Can the bot stand with its feet at `pos`? Either normally (clear
    /// cell over a solid or step floor) or inside a step cell (standing
    /// on the slab/stair, which raises the body half a block, so the
    /// two cells above must be clear). Lava remains occupiable here so an
    /// already-trapped bot can generate outward moves; the central planner's
    /// lava policy decides which of those transitions are legal.
    fn standable(&self, pos: BlockPos) -> bool {
        // where the body can be: air, or wading in lava
        let occupiable = |k: BlockKind| matches!(k, BlockKind::Air | BlockKind::Lava);
        // what can hold the body up for a *full-height* stand: solids or the
        // lava surface. A step (bottom slab/stair/carpet) is deliberately NOT
        // supportive here: standing on a step raises the body only half a
        // block and is modeled by the `Step` arm below. Treating the air cell
        // above a step as standable created a phantom foothold that let
        // JumpMove climb stairs as ordinary full blocks instead of walking
        // them (auto-step), producing wrong paths and a robotic hop-per-stair.
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

/// Classify a block state. Slabs/stairs/carpet are recognized by their
/// block-kind name plus orientation properties: only bottom variants are
/// steps; top and double variants act like full blocks.
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
    // Fences, cobblestone/etc. walls, and fence gates are 1.5 blocks tall — you
    // can't hop them or stand on top by stepping up from the side. Classifying
    // them as their own kind (NOT Solid) stops the planner routing up-and-over a
    // fence post (a route the bot can't execute → bump/jump/setback oscillation);
    // it routes around instead. ("*Wall" only matches real wall blocks; wall-
    // mounted decorations end with their own noun — Torch/Sign/Banner/Skull.)
    if name.ends_with("Fence") || name.ends_with("FenceGate") || name.ends_with("Wall") {
        return BlockKind::Fence;
    }
    // Water is left as Solid (impassable) for now — swimming isn't modeled.
    // Other thin blocks (panes, bars) read as Solid, which is safe: they're one
    // block tall, so the planner simply won't path through them.
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

/// An OWNED copy of a bounded world region — no lock, no lifetime.
///
/// Why this exists: A* over a live world view takes the world `RwLock` read
/// lock once per block query, in a tight loop, for up to the
/// whole search budget. parking_lot's RwLock lets that continuous read
/// pressure starve azalea's tick, which needs `world.write()` every tick for
/// chunk/entity updates — so a long search froze the entire client (all loops
/// stalled at once; the stall watchdog's "tick broadcaster froze" signature).
/// Wrapping the search in a `timeout` did NOT help: `spawn_blocking` can't be
/// cancelled, so the abandoned search kept holding the lock.
///
/// [`capture`](Self::capture) copies a cube around the start under ONE brief
/// read lock (block-state lookup is a cheap array index; classification is
/// memoized), then the search runs over this owned map holding NO lock — so it
/// can never starve the tick, however long it runs. Cells outside the captured
/// cube (or genuinely unloaded) read as [`BlockKind::Unloaded`], exactly as
/// before, so the incremental-leg re-planning in `walk_to` still walks toward
/// far goals and re-captures from the new position.
pub struct WorldSnapshot {
    /// Dense X/Z/Y array over `lo..=hi`. A bounded planning snapshot can exceed
    /// a million cells; one compact enum per cell is dramatically cheaper than
    /// a hash-map allocation per block and gives constant-time indexed reads.
    blocks: Vec<BlockKind>,
    lo: BlockPos,
    hi: BlockPos,
    size_y: usize,
    size_z: usize,
}

impl WorldSnapshot {
    /// Copy the inclusive AABB `lo..=hi` under one brief read lock. The caller
    /// sizes the box to CONTAIN the goal (plus margin) — a fixed cube around
    /// the start hid any goal or corridor beyond its radius, so the search
    /// couldn't see a straight path and detoured (over lava). Every cell is
    /// classified once into a compact dense array; missing chunk data remains
    /// `Unloaded` (impassable).
    pub fn capture(world: &parking_lot::RwLock<World>, lo: BlockPos, hi: BlockPos) -> Self {
        let size_x = (hi.x - lo.x + 1).max(0) as usize;
        let size_y = (hi.y - lo.y + 1).max(0) as usize;
        let size_z = (hi.z - lo.z + 1).max(0) as usize;
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
