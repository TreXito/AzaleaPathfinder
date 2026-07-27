//! Motion metadata carried alongside the compatibility [`crate::Path`].
//!
//! `Path` and `PathNode` intentionally remain unchanged for downstream source
//! compatibility. New adaptive and invalidation code uses [`MotionPlan`],
//! whose sidecar steps preserve generator semantics such as parkour distance
//! and the distinction between a vertical swim stroke and a bank exit.

use azalea::BlockPos;

use crate::adaptive::{CostComponents, MoveFeatures, PrimitiveId, TerrainClass};
use crate::local::moves::{MoveContext, SAFE_FALL};
use crate::local::world::{BlockKind, WorldView, offset};
use crate::{MoveKind, Path};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionStep {
    pub edge_index: u32,
    pub from: BlockPos,
    pub to: BlockPos,
    pub kind: MoveKind,
    pub primitive: PrimitiveId,
    pub features: MoveFeatures,
    pub predicted: CostComponents,
    /// Frozen nominal duration used only as the denominator for learning a
    /// dimensionless relative multiplier.
    pub predicted_ticks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionMetadata {
    pub primitive: PrimitiveId,
    pub features: MoveFeatures,
    pub predicted: CostComponents,
    pub predicted_ticks: u32,
}

#[derive(Debug, Clone)]
pub struct MotionPlan {
    path: Path,
    steps: Vec<MotionStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionPlanError {
    InvalidStartTransition,
    MetadataMismatch,
    MissingGeneratedMetadata,
}

impl std::fmt::Display for MotionPlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid motion plan: {self:?}")
    }
}

impl std::error::Error for MotionPlanError {}

impl MotionPlan {
    pub fn try_from_path(
        path: Path,
        world: &dyn WorldView,
        context: &MoveContext,
    ) -> Result<Self, MotionPlanError> {
        let steps: Result<Vec<_>, _> = path
            .nodes
            .windows(2)
            .enumerate()
            .map(|(edge_index, nodes)| {
                if nodes[1].reached_by == MoveKind::Start {
                    return Err(MotionPlanError::InvalidStartTransition);
                }
                Ok(describe_builtin_step(
                    edge_index as u32,
                    nodes[0].pos,
                    nodes[1].pos,
                    nodes[1].reached_by,
                    world,
                    context,
                ))
            })
            .collect();
        Ok(Self {
            path,
            steps: steps?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn steps(&self) -> &[MotionStep] {
        &self.steps
    }

    pub fn into_path(self) -> Path {
        self.path
    }

    pub fn into_parts(self) -> (Path, Vec<MotionStep>) {
        (self.path, self.steps)
    }

    pub(crate) fn from_parts(path: Path, steps: Vec<MotionStep>) -> Result<Self, MotionPlanError> {
        if steps.len() != path.nodes.len().saturating_sub(1)
            || steps.iter().enumerate().any(|(index, step)| {
                step.edge_index != index as u32
                    || path.nodes[index].pos != step.from
                    || path.nodes[index + 1].pos != step.to
                    || path.nodes[index + 1].reached_by != step.kind
            })
        {
            return Err(MotionPlanError::MetadataMismatch);
        }
        Ok(Self { path, steps })
    }
}

fn describe_builtin_step(
    edge_index: u32,
    from: BlockPos,
    to: BlockPos,
    kind: MoveKind,
    world: &dyn WorldView,
    context: &MoveContext,
) -> MotionStep {
    let bank_exit = kind == MoveKind::Swim
        && world.block(from) == BlockKind::Water
        && world.block(to) != BlockKind::Water
        && to.y > from.y;
    let primitive = match kind {
        MoveKind::Start => unreachable!("a transition cannot be reached by Start"),
        MoveKind::Walk if from.y != to.y => PrimitiveId::STEP,
        MoveKind::Walk if from.x != to.x && from.z != to.z => PrimitiveId::WALK_DIAGONAL,
        MoveKind::Walk => PrimitiveId::WALK_CARDINAL,
        MoveKind::Jump => PrimitiveId::JUMP,
        MoveKind::Fall => PrimitiveId::FALL,
        MoveKind::Parkour { .. } => PrimitiveId::PARKOUR,
        MoveKind::Climb => PrimitiveId::CLIMB,
        MoveKind::Swim if bank_exit => PrimitiveId::SWIM_EXIT,
        MoveKind::Swim => PrimitiveId::SWIM,
        MoveKind::Aotv => PrimitiveId::AOTV,
        MoveKind::Etherwarp => PrimitiveId::ETHERWARP,
    };
    let baseline = legacy_baseline_cost(kind, primitive, from, to, world, context);
    let metadata =
        metadata_from_generated_edge(primitive, from, to, kind, world, context, baseline);
    MotionStep {
        edge_index,
        from,
        to,
        kind,
        primitive: metadata.primitive,
        features: metadata.features,
        predicted: metadata.predicted,
        predicted_ticks: metadata.predicted_ticks,
    }
}

pub(crate) fn metadata_from_generated_edge(
    primitive: PrimitiveId,
    from: BlockPos,
    to: BlockPos,
    kind: MoveKind,
    world: &dyn WorldView,
    context: &MoveContext,
    generated_cost: crate::Cost,
) -> MotionMetadata {
    let horizontal_blocks = match kind {
        MoveKind::Parkour { blocks, .. } => blocks.min(16),
        _ => from.x.abs_diff(to.x).max(from.z.abs_diff(to.z)).min(16) as u8,
    };
    let vertical = i64::from(to.y)
        .saturating_sub(i64::from(from.y))
        .clamp(i64::from(i8::MIN), i64::from(i8::MAX)) as i8;
    let destination = world.block(to);
    let surface = match destination {
        // The feet occupy these blocks; sampling below would erase the
        // movement regime (notably water landings and slab/stair steps).
        BlockKind::Step(_) | BlockKind::Water | BlockKind::Climbable | BlockKind::Lava => {
            destination
        }
        _ => world.block(offset(to, 0, -1, 0)),
    };
    let terrain = terrain_class(surface);
    let adjacent_walls = adjacent_wall_count(world, to);
    let mut flags = 0;
    if !matches!(
        world.block(offset(to, 0, 2, 0)),
        BlockKind::Air | BlockKind::Passable | BlockKind::Water
    ) {
        flags |= MoveFeatures::FLAG_LOW_HEADROOM;
    }
    if fractional_neighbor(world, to) {
        flags |= MoveFeatures::FLAG_FRACTIONAL_NEIGHBOR;
    }
    if primitive == PrimitiveId::SWIM
        && world.block(from) != BlockKind::Water
        && world.block(to) == BlockKind::Water
    {
        flags |= MoveFeatures::FLAG_WATER_ENTRY;
    }
    let features = MoveFeatures::new(horizontal_blocks, vertical, terrain)
        .with_surroundings(adjacent_walls, flags);
    let drop = from.y.saturating_sub(to.y).max(0) as u32;
    let hurt = drop.saturating_sub(SAFE_FALL as u32);
    let requested_damage = if matches!(primitive, PrimitiveId::FALL | PrimitiveId::PARKOUR)
        && world.block(to) != BlockKind::Water
    {
        context.fall_damage_penalty.saturating_mul(hurt)
    } else {
        0
    };
    let requested_safety = if primitive == PrimitiveId::SWIM && world.block(to) == BlockKind::Water
    {
        context.water_penalty.min(generated_cost)
    } else {
        0
    };
    // Generator cost is authoritative. Fixed policy/safety is allocated first,
    // then unavoidable damage, and only the exact remainder is adaptable
    // movement time. This guarantees metadata can never make A* pay more or
    // less than the edge it describes, even if a custom caller supplies an
    // internally inconsistent edge.
    let safety = requested_safety.min(generated_cost);
    let after_safety = generated_cost.saturating_sub(safety);
    let damage = requested_damage.min(after_safety);
    let predicted = CostComponents {
        time: after_safety.saturating_sub(damage),
        safety,
        damage,
        ..CostComponents::default()
    };
    debug_assert_eq!(predicted.total(), generated_cost);
    let predicted_ticks = nominal_ticks(kind, primitive, from, to, world);
    MotionMetadata {
        primitive,
        features,
        predicted,
        predicted_ticks,
    }
}

fn terrain_class(block: BlockKind) -> TerrainClass {
    match block {
        BlockKind::Step(_) => TerrainClass::PartialBlock,
        BlockKind::Climbable => TerrainClass::Climbable,
        BlockKind::Water => TerrainClass::Water,
        BlockKind::Air | BlockKind::Passable | BlockKind::Lava | BlockKind::Unloaded => {
            TerrainClass::Unknown
        }
        BlockKind::Solid | BlockKind::Fence => TerrainClass::FullBlock,
    }
}

fn adjacent_wall_count(world: &dyn WorldView, position: BlockPos) -> u8 {
    [(1, 0), (-1, 0), (0, 1), (0, -1)]
        .into_iter()
        .filter(|(dx, dz)| {
            let feet = world.block(offset(position, *dx, 0, *dz));
            let head = world.block(offset(position, *dx, 1, *dz));
            !matches!(
                feet,
                BlockKind::Air | BlockKind::Passable | BlockKind::Water
            ) || !matches!(
                head,
                BlockKind::Air | BlockKind::Passable | BlockKind::Water
            )
        })
        .count() as u8
}

fn fractional_neighbor(world: &dyn WorldView, position: BlockPos) -> bool {
    (-1..=1).any(|dx| {
        (-1..=1).any(|dz| {
            (dx, dz) != (0, 0)
                && matches!(world.block(offset(position, dx, 0, dz)), BlockKind::Step(_))
        })
    })
}

fn legacy_baseline_cost(
    kind: MoveKind,
    primitive: PrimitiveId,
    from: BlockPos,
    to: BlockPos,
    world: &dyn WorldView,
    context: &MoveContext,
) -> crate::Cost {
    let horizontal = from.x.abs_diff(to.x).max(from.z.abs_diff(to.z)).max(1);
    let drop = from.y.saturating_sub(to.y).max(0) as u32;
    let hurt = drop.saturating_sub(SAFE_FALL as u32);
    let time = match primitive {
        PrimitiveId::WALK_CARDINAL => context.costs.cardinal_walk,
        PrimitiveId::WALK_DIAGONAL => context.costs.diagonal_walk,
        PrimitiveId::STEP => context.costs.step,
        PrimitiveId::JUMP => {
            if matches!(world.block(to), BlockKind::Step(_)) {
                context.costs.step
            } else {
                context.costs.jump
            }
        }
        PrimitiveId::FALL => context
            .costs
            .fall_base
            .saturating_add(context.costs.fall_per_block.saturating_mul(drop)),
        PrimitiveId::PARKOUR => {
            let blocks = match kind {
                MoveKind::Parkour { blocks, .. } => u32::from(blocks),
                _ => horizontal,
            };
            context
                .costs
                .parkour_per_block
                .saturating_mul(blocks)
                .saturating_add(if to.y > from.y {
                    context.costs.jump.saturating_mul(to.y.abs_diff(from.y))
                } else {
                    context.costs.fall_per_block.saturating_mul(drop)
                })
        }
        PrimitiveId::CLIMB => {
            if is_high_climb_grab(from, to, world) {
                context
                    .costs
                    .climb
                    .saturating_add(context.costs.climb)
                    .saturating_add(context.costs.jump)
            } else {
                context.costs.climb
            }
        }
        PrimitiveId::SWIM => {
            context
                .costs
                .swim
                .saturating_add(if world.block(to) == BlockKind::Water {
                    context.water_penalty
                } else {
                    0
                })
        }
        PrimitiveId::SWIM_EXIT => context.costs.swim_exit,
        // These extension points currently produce no built-in edges. Keep a
        // non-zero placeholder so custom executors can still record them.
        PrimitiveId::AOTV | PrimitiveId::ETHERWARP => context.costs.cardinal_walk,
        _ => context.costs.cardinal_walk,
    };
    let damage = if matches!(primitive, PrimitiveId::FALL | PrimitiveId::PARKOUR)
        && world.block(to) != BlockKind::Water
    {
        context.fall_damage_penalty.saturating_mul(hurt)
    } else {
        0
    };
    time.saturating_add(damage)
}

fn is_high_climb_grab(from: BlockPos, to: BlockPos, world: &dyn WorldView) -> bool {
    world.block(from) != BlockKind::Climbable
        && world.block(to) == BlockKind::Climbable
        && to.y > from.y
        && (from.x != to.x || from.z != to.z)
}

fn nominal_ticks(
    kind: MoveKind,
    primitive: PrimitiveId,
    from: BlockPos,
    to: BlockPos,
    world: &dyn WorldView,
) -> u32 {
    let drop = from.y.saturating_sub(to.y).max(0) as u32;
    match primitive {
        PrimitiveId::WALK_CARDINAL => 5,
        PrimitiveId::WALK_DIAGONAL => 7,
        PrimitiveId::STEP => 7,
        PrimitiveId::JUMP => 11,
        PrimitiveId::FALL => 4_u32.saturating_add(drop.saturating_mul(2)),
        PrimitiveId::PARKOUR => {
            let blocks = match kind {
                MoveKind::Parkour { blocks, .. } => u32::from(blocks),
                _ => from.x.abs_diff(to.x).max(from.z.abs_diff(to.z)),
            };
            blocks.saturating_mul(6).max(1)
        }
        PrimitiveId::CLIMB if is_high_climb_grab(from, to, world) => {
            // The generator prices this as two rung-equivalents plus a jump,
            // rather than an ordinary one-rung climb. Keep the learning
            // denominator composed from the same physical actions.
            15_u32.saturating_mul(2).saturating_add(11)
        }
        PrimitiveId::CLIMB => 15_u32.saturating_mul(from.y.abs_diff(to.y).max(1)),
        PrimitiveId::SWIM => 10,
        PrimitiveId::SWIM_EXIT => 20,
        PrimitiveId::AOTV | PrimitiveId::ETHERWARP => 10,
        _ => 10,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{PathNode, WorldView};

    struct Grid(HashMap<(i32, i32, i32), BlockKind>);

    impl WorldView for Grid {
        fn block(&self, position: BlockPos) -> BlockKind {
            self.0
                .get(&(position.x, position.y, position.z))
                .copied()
                .unwrap_or(BlockKind::Air)
        }
    }

    fn path(from: BlockPos, to: BlockPos, kind: MoveKind) -> Path {
        Path {
            nodes: vec![
                PathNode {
                    pos: from,
                    reached_by: MoveKind::Start,
                },
                PathNode {
                    pos: to,
                    reached_by: kind,
                },
            ],
            total_cost: 10,
        }
    }

    #[test]
    fn vertical_swim_is_not_a_bank_exit() {
        let from = BlockPos::new(0, 64, 0);
        let to = BlockPos::new(0, 65, 0);
        let world = Grid(
            [
                ((0, 64, 0), BlockKind::Water),
                ((0, 65, 0), BlockKind::Water),
            ]
            .into_iter()
            .collect(),
        );
        let plan = MotionPlan::try_from_path(
            path(from, to, MoveKind::Swim),
            &world,
            &MoveContext::default(),
        )
        .unwrap();
        assert_eq!(plan.steps()[0].primitive, PrimitiveId::SWIM);
    }

    #[test]
    fn raised_dry_bank_is_an_exit() {
        let from = BlockPos::new(0, 64, 0);
        let to = BlockPos::new(1, 65, 0);
        let world = Grid(
            [((0, 64, 0), BlockKind::Water), ((1, 65, 0), BlockKind::Air)]
                .into_iter()
                .collect(),
        );
        let plan = MotionPlan::try_from_path(
            path(from, to, MoveKind::Swim),
            &world,
            &MoveContext::default(),
        )
        .unwrap();
        assert_eq!(plan.steps()[0].primitive, PrimitiveId::SWIM_EXIT);
    }

    #[test]
    fn diagonal_parkour_uses_generator_block_count() {
        let plan = MotionPlan::try_from_path(
            path(
                BlockPos::new(0, 64, 0),
                BlockPos::new(2, 64, 2),
                MoveKind::Parkour { blocks: 3, rise: 0 },
            ),
            &Grid(HashMap::new()),
            &MoveContext::default(),
        )
        .unwrap();
        assert_eq!(plan.steps()[0].features.horizontal_blocks, 3);
        assert_eq!(plan.steps()[0].predicted.time, 66);
    }

    #[test]
    fn fall_keeps_base_slope_and_damage_separate() {
        let plan = MotionPlan::try_from_path(
            path(
                BlockPos::new(0, 70, 0),
                BlockPos::new(1, 65, 0),
                MoveKind::Fall,
            ),
            &Grid(HashMap::new()),
            &MoveContext::default(),
        )
        .unwrap();
        let step = &plan.steps()[0];
        assert_eq!(
            step.predicted.time,
            MoveContext::default().costs.fall_base
                + MoveContext::default().costs.fall_per_block * 5
        );
        assert_eq!(
            step.predicted.damage,
            MoveContext::default().fall_damage_penalty * 2
        );
        assert_eq!(step.predicted.total(), 134);
    }

    #[test]
    fn fallback_metadata_preserves_high_climb_and_rising_parkour_costs() {
        let context = MoveContext::default();
        let climb_from = BlockPos::new(0, 64, 0);
        let climb_to = BlockPos::new(1, 65, 0);
        let climb_world = Grid([((1, 65, 0), BlockKind::Climbable)].into_iter().collect());
        let climb = MotionPlan::try_from_path(
            path(climb_from, climb_to, MoveKind::Climb),
            &climb_world,
            &context,
        )
        .unwrap();
        assert_eq!(
            climb.steps()[0].predicted.total(),
            context
                .costs
                .climb
                .saturating_mul(2)
                .saturating_add(context.costs.jump)
        );
        assert_eq!(climb.steps()[0].predicted_ticks, 41);

        let parkour = MotionPlan::try_from_path(
            path(
                BlockPos::new(0, 64, 0),
                BlockPos::new(3, 65, 0),
                MoveKind::Parkour { blocks: 3, rise: 1 },
            ),
            &Grid(HashMap::new()),
            &context,
        )
        .unwrap();
        assert_eq!(
            parkour.steps()[0].predicted.total(),
            context
                .costs
                .parkour_per_block
                .saturating_mul(3)
                .saturating_add(context.costs.jump)
        );
    }

    #[test]
    fn water_toll_is_fixed_safety_and_water_fall_has_no_damage() {
        let context = MoveContext::default();
        let swim_from = BlockPos::new(0, 64, 0);
        let swim_to = BlockPos::new(1, 64, 0);
        let swim_world = Grid(
            [
                ((0, 64, 0), BlockKind::Water),
                ((1, 64, 0), BlockKind::Water),
            ]
            .into_iter()
            .collect(),
        );
        let swim = MotionPlan::try_from_path(
            path(swim_from, swim_to, MoveKind::Swim),
            &swim_world,
            &context,
        )
        .unwrap();
        let swim = &swim.steps()[0];
        assert_eq!(swim.predicted.time, context.costs.swim);
        assert_eq!(swim.predicted.safety, context.water_penalty);
        assert_eq!(swim.features.flags & MoveFeatures::FLAG_WATER_ENTRY, 0);
        assert_eq!(
            swim.predicted.total(),
            context.costs.swim.saturating_add(context.water_penalty)
        );

        let entry_from = BlockPos::new(-1, 64, 0);
        let entry_world = Grid([((0, 64, 0), BlockKind::Water)].into_iter().collect());
        let entry = MotionPlan::try_from_path(
            path(entry_from, swim_from, MoveKind::Swim),
            &entry_world,
            &context,
        )
        .unwrap();
        let entry = &entry.steps()[0];
        assert_ne!(entry.features.flags & MoveFeatures::FLAG_WATER_ENTRY, 0);
        assert_ne!(entry.features, swim.features);
        assert_eq!(entry.predicted, swim.predicted);

        let fall_from = BlockPos::new(0, 70, 0);
        let fall_to = BlockPos::new(1, 65, 0);
        let fall_world = Grid([((1, 65, 0), BlockKind::Water)].into_iter().collect());
        let fall = MotionPlan::try_from_path(
            path(fall_from, fall_to, MoveKind::Fall),
            &fall_world,
            &context,
        )
        .unwrap();
        let fall = &fall.steps()[0];
        assert_eq!(fall.predicted.damage, 0);
        assert_eq!(
            fall.predicted.time,
            context
                .costs
                .fall_base
                .saturating_add(context.costs.fall_per_block.saturating_mul(5))
        );
        assert_eq!(fall.features.terrain, TerrainClass::Water);
    }

    #[test]
    fn component_split_never_exceeds_authoritative_edge_cost() {
        let from = BlockPos::new(0, 70, 0);
        let to = BlockPos::new(1, 60, 0);
        let context = MoveContext {
            fall_damage_penalty: crate::Cost::MAX,
            water_penalty: crate::Cost::MAX,
            ..MoveContext::default()
        };
        let metadata = metadata_from_generated_edge(
            PrimitiveId::FALL,
            from,
            to,
            MoveKind::Fall,
            &Grid(HashMap::new()),
            &context,
            5,
        );
        assert_eq!(metadata.predicted.total(), 5);
        assert_eq!(metadata.predicted.damage, 5);
        assert_eq!(metadata.predicted.time, 0);

        let water = Grid([((1, 60, 0), BlockKind::Water)].into_iter().collect());
        let metadata = metadata_from_generated_edge(
            PrimitiveId::SWIM,
            from,
            to,
            MoveKind::Swim,
            &water,
            &context,
            7,
        );
        assert_eq!(metadata.predicted.total(), 7);
        assert_eq!(metadata.predicted.safety, 7);
        assert_eq!(metadata.predicted.time, 0);
    }
}
