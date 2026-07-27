use std::collections::HashMap;

use azalea::BlockPos;
use azalea::block::fluid_state::{FluidKind, FluidState};
use azalea::block::properties::{SlabKind, TopBottom};
use azalea::block::{BlockState, Property};
use azalea::physics::collision::BlockWithShape;
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
    /// A bottom slab, bottom stair, carpet or snow layer that can be walked
    /// onto, carrying its collision height in sixteenths of a block.
    ///
    /// The height is not decoration. A body is 0.6 wide and stands on the
    /// tallest thing any part of it overlaps, so a bot walking beside a snow
    /// layer two layers deeper than the one under its feet gets lifted by a
    /// corner of its own hitbox - and the server and the client do not always
    /// resolve that same corner the same way. Measured on the hub: every
    /// anticheat Simulation flag in a summit climb landed on a step-up onto a
    /// fractional height, always off by exactly a quarter of a block, with both
    /// sides holding identical block data. Without the height here the planner
    /// cannot tell a flat field of snow from a staircase of it.
    Step(u8),
    /// A block with no collision at all: grass, flowers, crops, torches, signs,
    /// rails, ladders, thin snow. The body walks straight through it, so it is
    /// empty space that happens to have a name.
    ///
    /// This is not a nicety. Classifying these `Solid` deleted 5% of this map's
    /// standable floor, and 27% of the deleted tiles sat in clusters of four or
    /// more, so whole grass fields and crop farms became walls the planner
    /// would not cross while the bot stood in the middle of one.
    Passable,
    /// A ladder, vine or scaffolding. The body passes through it and can hold
    /// position anywhere on it without anything underneath, which is the whole
    /// point: it is the only move in the game that gains height without a
    /// jump, and on a hand-built map it is usually the only way up a face.
    Climbable,
    /// Lava. It remains occupiable so a trapped bot can plan an exit.
    Lava,
    /// Water, including waterlogged blocks the body can pass through. The
    /// player swims here: it neither supports weight nor makes the player fall.
    Water,
    /// Missing chunk data, treated as impassable.
    Unloaded,
}

/// The block data needed by the local planner.
pub trait WorldView {
    fn block(&self, pos: BlockPos) -> BlockKind;

    /// Whether this world holds any lava at all.
    ///
    /// The lava rules are the most expensive thing the search does per node by
    /// a wide margin, and on most maps they are answering a question with no
    /// lava in it. Clearing a node costs a 5x5x3 exposure scan, and the soft
    /// avoidance ring costs another 240 block reads, so a node that is nowhere
    /// near lava still pays about 315 reads to establish that - several times
    /// what generating its moves costs. Measured on a 240 block synthetic
    /// route, skipping both when the region holds no lava cuts the search from
    /// 1190ms to 356ms, and that time is not a nicety: a leg reaches as far as
    /// one time budget can search.
    ///
    /// Answering `true` is always safe, so the default is the conservative one
    /// and only a view that has already seen every block it will ever be asked
    /// about overrides it. [`WorldSnapshot`] is exactly that, and it learns the
    /// answer during the copy it was making anyway.
    fn may_contain_lava(&self) -> bool {
        true
    }

    /// Returns whether the bot can hold position with its feet at `pos`, either
    /// standing on something or floating in water.
    /// Lava is occupiable here; the movement policy decides whether entry is safe.
    fn standable(&self, pos: BlockPos) -> bool {
        let occupiable = |k: BlockKind| {
            matches!(
                k,
                BlockKind::Air
                    | BlockKind::Passable
                    | BlockKind::Climbable
                    | BlockKind::Lava
                    | BlockKind::Water
            )
        };
        // Steps are separate because they raise the body half a block.
        // A ladder is deliberately not supportive: its collision box is a
        // sliver against the wall, so there is nothing to stand on top of.
        let supportive = |k: BlockKind| matches!(k, BlockKind::Solid | BlockKind::Lava);
        match self.block(pos) {
            BlockKind::Air | BlockKind::Passable | BlockKind::Lava => {
                occupiable(self.block(offset(pos, 0, 1, 0)))
                    && supportive(self.block(offset(pos, 0, -1, 0)))
            }
            // A climber hangs on the ladder itself, so like water the block
            // below is not consulted. Head room is the whole requirement.
            BlockKind::Climbable => occupiable(self.block(offset(pos, 0, 1, 0))),
            // A swimmer is held up by the water itself, so the block below is
            // deliberately not consulted: a player in water neither stands nor
            // falls, and requiring a floor would rule out every column of open
            // water. Room for the body is the whole requirement.
            BlockKind::Water => occupiable(self.block(offset(pos, 0, 1, 0))),
            // Air or a plant only: a slab under water is a swim, and SwimMove
            // owns that. Widening this to every occupiable kind would let the
            // walker plan along a flooded floor it cannot actually walk.
            BlockKind::Step(_) => {
                let open = |k: BlockKind| matches!(k, BlockKind::Air | BlockKind::Passable);
                open(self.block(offset(pos, 0, 1, 0))) && open(self.block(offset(pos, 0, 2, 0)))
            }
            _ => false,
        }
    }
}

/// A step exactly half a block tall, which is what a slab and a bottom stair
/// are. Named because those two are measured by name rather than by geometry: a
/// stair's bounding box is a full cube, since the box wraps the whole L.
pub const HALF_BLOCK: u8 = 8;

/// A collision height in sixteenths, which is the unit every partial block in
/// the game is actually built from: snow gains two per layer, a carpet is one,
/// a slab is eight.
fn sixteenths(height: f64) -> u8 {
    (height * 16.0).round().clamp(0.0, 16.0) as u8
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
    // A bubble column drags the player up or down with a velocity nothing here
    // models, so it stays impassable rather than becoming a swim the bot cannot
    // hold position in.
    if name == "BubbleColumn" {
        return BlockKind::Solid;
    }
    // Asked before the shape tests because a ladder's collision box is not
    // empty: it is a sliver against the wall, so the shape alone says "Solid",
    // which turns the one route up a sheer face into a wall. The tag is the
    // same one azalea's own physics uses to decide `OnClimbable`, so the
    // planner and the client agree on what is a ladder by construction.
    if azalea::registry::tags::blocks::CLIMBABLE.contains(&AzaleaBlockKind::from(state)) {
        return BlockKind::Climbable;
    }
    // Waterlogging is a property of the block rather than a block of its own,
    // so the fluid state is what catches waterlogged signs and ladders as well
    // as plain water, kelp and seagrass. The collision shape decides which of
    // the two halves wins: a waterlogged slab is still a slab to walk on, and
    // only blocks the body can pass through are swum.
    if FluidState::from(state).kind == FluidKind::Water && state.is_collision_shape_empty() {
        return BlockKind::Water;
    }
    // No collision box at all means the body passes straight through, so this
    // is empty space with a name on it: grass, flowers, crops, torches, signs,
    // buttons, rails, ladders, a single layer of snow. Asked before the shape
    // tests below because none of those have an empty shape, so the order is
    // free, and asked from the block state rather than a name list so a block
    // nobody thought of still gets the right answer.
    if state.is_collision_shape_empty() {
        return BlockKind::Passable;
    }
    // Anything whose collision box tops out at or below half a block is stood
    // *inside* rather than on top of: the feet sit at y + 0.09 on a lily pad,
    // y + 0.5 on a slab. The named cases below cover slabs, stairs and carpets;
    // this catches the rest by geometry, which is how a lily pad should have
    // been handled all along. Classified `Solid`, a lily pad made the block the
    // bot was physically standing in un-standable to its own planner, so every
    // plan from a lily pad returned a path of one node: a bot marooned on a
    // lake with water forbidden, reporting Failed seven legs later.
    //
    // `collision_shape` is in block-LOCAL 0..1 coordinates, so the position
    // passed in is irrelevant to the height and 0,0,0 keeps that explicit.
    let top = state.collision_shape(BlockPos::new(0, 0, 0)).bounds().max.y;
    if top <= 0.5 {
        return BlockKind::Step(sixteenths(top));
    }
    if name.ends_with("Slab") {
        return match SlabKind::try_from_block_state(state) {
            Some(SlabKind::Bottom) => BlockKind::Step(HALF_BLOCK),
            _ => BlockKind::Solid,
        };
    }
    if name.ends_with("Stairs") {
        return match TopBottom::try_from_block_state(state) {
            Some(TopBottom::Bottom) => BlockKind::Step(HALF_BLOCK),
            _ => BlockKind::Solid,
        };
    }
    if name.ends_with("Carpet") {
        return BlockKind::Step(sixteenths(
            state.collision_shape(BlockPos::new(0, 0, 0)).bounds().max.y,
        ));
    }
    // These are taller than a full block and cannot be climbed from the side.
    if name.ends_with("Fence") || name.ends_with("FenceGate") || name.ends_with("Wall") {
        return BlockKind::Fence;
    }
    // Thin-block geometry is not modeled yet.
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
    /// Whether any captured cell is lava. See [`WorldView::may_contain_lava`].
    has_lava: bool,
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
        let mut has_lava = false;
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
                    has_lava |= kind == BlockKind::Lava;
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
            has_lava,
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

    fn may_contain_lava(&self) -> bool {
        self.has_lava
    }
}

#[cfg(test)]
mod classify_tests {
    use azalea::block::blocks;
    use azalea::block::properties::WaterLevel;

    use super::*;

    #[test]
    fn water_is_swimmable_but_waterlogged_geometry_keeps_its_shape() {
        assert_eq!(
            classify_state(BlockState::from(blocks::Water {
                level: WaterLevel::_0
            })),
            BlockKind::Water
        );
        assert_eq!(
            classify_state(BlockState::from(blocks::KelpPlant)),
            BlockKind::Water
        );
        // A waterlogged slab is still a slab: the body cannot pass through it,
        // so it stays something to walk onto rather than something to swim in.
        assert_eq!(
            classify_state(BlockState::from(blocks::OakSlab {
                kind: SlabKind::Bottom,
                waterlogged: true,
            })),
            BlockKind::Step(HALF_BLOCK)
        );
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
            has_lava: false,
        };

        assert_eq!(snapshot.block(BlockPos::new(0, 63, 0)), BlockKind::Unloaded);
        assert_eq!(snapshot.block(BlockPos::new(0, 64, 0)), BlockKind::Solid);
        assert_eq!(snapshot.block(BlockPos::new(0, 65, 0)), BlockKind::Air);
        assert_eq!(snapshot.block(BlockPos::new(1, 64, 0)), BlockKind::Unloaded);
        assert_eq!(snapshot.block(BlockPos::new(0, 62, 0)), BlockKind::Unloaded);
    }
}


#[cfg(test)]
mod snow_tests {
    use azalea::block::blocks;
    use azalea::block::properties::Layers;

    use super::*;

    fn snow(layers: u32) -> BlockState {
        BlockState::from(blocks::Snow {
            layers: match layers {
                1 => Layers::_1,
                2 => Layers::_2,
                3 => Layers::_3,
                4 => Layers::_4,
                5 => Layers::_5,
                6 => Layers::_6,
                7 => Layers::_7,
                _ => Layers::_8,
            },
        })
    }

    /// Snow is a *lagging* stack: its collision box is one layer shorter than
    /// the layer count suggests.
    ///
    /// Vanilla builds `SHAPE_BY_LAYER[i] = box(0, 0, 0, 16, i * 2, 16)` and
    /// then `getCollisionShape` returns `SHAPE_BY_LAYER[layers - 1]`, while the
    /// *outline* uses `[layers]`. So the box you can see is always one layer
    /// taller than the box you stand on, one layer of snow holds nothing up at
    /// all, and even eight layers - a block that looks completely full - is
    /// only 0.875 tall.
    ///
    /// This is recorded because it is the obvious thing to get wrong: the
    /// intuitive `layers / 8` is wrong for every value, and the error is small
    /// enough (an eighth of a block) to look like drift rather than a bug. It
    /// was checked here against a real anticheat desync on snow. Azalea is
    /// right; the guess was wrong.
    #[test]
    fn snow_layer_collision_lags_one_layer_behind() {
        // A single layer really has *no* box, not a zero-height one, and
        // asking an empty shape for its bounds panics. That is also why
        // `classify_state` tests emptiness before it measures anything.
        assert!(snow(1).is_collision_shape_empty(), "snow[layers=1] has a box");
        for layers in 2..=8u32 {
            let got = snow(layers)
                .collision_shape(BlockPos::new(0, 0, 0))
                .bounds()
                .max
                .y;
            let want = f64::from(layers - 1) / 8.0;
            assert!(
                (got - want).abs() < 1e-9,
                "snow[layers={layers}]: azalea says {got}, vanilla says {want}"
            );
        }
    }

    /// Where the planner's half-block cutoff falls on that stack.
    ///
    /// Anything half a block or shorter is a `Step`, which the body walks up
    /// without jumping; taller is `Solid` and has to be jumped, because the
    /// player's step height is 0.6. With the lagging box above, that line falls
    /// between five and six layers: five layers collide at exactly 0.5 and six
    /// at 0.625, which is past what a step can take.
    #[test]
    fn snow_is_stepped_to_five_layers_and_jumped_above() {
        assert_eq!(classify_state(snow(1)), BlockKind::Passable);
        // And the height comes with it: snow gains two sixteenths per layer,
        // one layer behind the count, so 2 layers collide at 1/16 and 5 at 4/8.
        for layers in 2..=5u32 {
            assert_eq!(
                classify_state(snow(layers)),
                BlockKind::Step(2 * (layers - 1) as u8),
                "{layers}"
            );
        }
        for layers in 6..=8u32 {
            assert_eq!(classify_state(snow(layers)), BlockKind::Solid, "{layers}");
        }
    }
}
