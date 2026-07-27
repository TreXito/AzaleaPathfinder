//! Route resilience primitives.
//!
//! This module deliberately implements conservative invalidation and bounded,
//! scoped obstacle memory rather than pretending that a restarted or patched
//! A* route is D* Lite.  A live movement validator remains the final authority:
//! snapshots and dependency footprints describe what was observed while a
//! route was planned, not an atomic view of a changing server.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use azalea::BlockPos;

use crate::local::moves::{LavaPolicy, MAX_LAVA_SCAN_RADIUS, MoveContext};
use crate::local::world::{BlockKind, WorldView, offset};
use crate::types::{Cost, MoveKind, Path, PathNode};

/// Maximum number of dynamic obstacles retained by one overlay.
pub const MAX_OVERLAY_ENTRIES: usize = 16_384;

/// Maximum number of block dependencies retained for one path edge.
///
/// If a conservative footprint exceeds this bound, the edge is marked broad
/// and any block update invalidates it.  Losing precision is safe; allocating
/// without a bound is not.
pub const MAX_DEPENDENCIES_PER_EDGE: usize = 32_768;

/// Maximum number of dependency-index entries retained for one path.
pub const MAX_INDEXED_DEPENDENCIES: usize = 1_000_000;

const MAX_SCOPE_COMPONENT_BYTES: usize = 128;
const MAX_SAMPLED_TRANSITION_LENGTH: i64 = 128;

/// Explicit identity of the world and dimension an observation belongs to.
///
/// Callers should use stable server-provided identifiers.  In particular,
/// entity positions from one dimension must never be reused in another merely
/// because their block coordinates happen to match.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorldScope {
    world: Box<str>,
    dimension: Box<str>,
}

impl WorldScope {
    /// Constructs a bounded, non-empty scope key.
    pub fn try_new(
        world: impl Into<String>,
        dimension: impl Into<String>,
    ) -> Result<Self, WorldScopeError> {
        let world = world.into();
        let dimension = dimension.into();
        validate_scope_component("world", &world)?;
        validate_scope_component("dimension", &dimension)?;
        Ok(Self {
            world: world.into_boxed_str(),
            dimension: dimension.into_boxed_str(),
        })
    }

    pub fn world(&self) -> &str {
        &self.world
    }

    pub fn dimension(&self) -> &str {
        &self.dimension
    }
}

fn validate_scope_component(field: &'static str, value: &str) -> Result<(), WorldScopeError> {
    if value.is_empty() {
        return Err(WorldScopeError::Empty(field));
    }
    if value.len() > MAX_SCOPE_COMPONENT_BYTES {
        return Err(WorldScopeError::TooLong {
            field,
            bytes: value.len(),
            maximum: MAX_SCOPE_COMPONENT_BYTES,
        });
    }
    if value.chars().any(char::is_control) {
        return Err(WorldScopeError::ContainsControl(field));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldScopeError {
    Empty(&'static str),
    TooLong {
        field: &'static str,
        bytes: usize,
        maximum: usize,
    },
    ContainsControl(&'static str),
}

impl fmt::Display for WorldScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(field) => write!(f, "{field} scope component is empty"),
            Self::TooLong {
                field,
                bytes,
                maximum,
            } => write!(
                f,
                "{field} scope component is {bytes} bytes; maximum is {maximum}"
            ),
            Self::ContainsControl(field) => {
                write!(f, "{field} scope component contains a control character")
            }
        }
    }
}

impl std::error::Error for WorldScopeError {}

/// Higher-priority observations survive capacity pressure and replace lower
/// priority observations at the same position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObstaclePriority(u8);

impl ObstaclePriority {
    pub const BACKGROUND: Self = Self(32);
    pub const NORMAL: Self = Self(128);
    pub const CRITICAL: Self = Self(255);

    pub const fn new(value: u8) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// How an active dynamic obstacle affects planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObstacleKind {
    /// Treat the occupied body block as non-supportive and impassable.
    Hard,
    /// Add a finite toll to routes that enter the position.
    Soft { penalty: Cost },
}

/// Stable identity of the bot or subsystem that owns an obstacle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObstacleOwner(pub u64);

impl ObstacleOwner {
    /// Reservations not owned by a particular bot.
    pub const SYSTEM: Self = Self(0);
}

/// Deterministic, bounded escape from mutual hard reservations.
///
/// Lower owner IDs have right of way: they see a higher-ID owner's hard
/// reservation as a soft toll, while the higher-ID bot yields.  Every hard
/// reservation also degrades to a soft toll after `max_hard_ticks`, preventing
/// a crashed owner from creating a permanent deadlock before its TTL expires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeadlockPolicy {
    pub max_hard_ticks: u64,
    pub yielded_soft_penalty: Cost,
}

impl Default for DeadlockPolicy {
    fn default() -> Self {
        Self {
            max_hard_ticks: 40,
            yielded_soft_penalty: 2_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObstacleEntry {
    owner: ObstacleOwner,
    kind: ObstacleKind,
    priority: ObstaclePriority,
    created_at_tick: u64,
    expires_at_tick: u64,
    sequence: u64,
}

impl ObstacleEntry {
    pub fn owner(&self) -> ObstacleOwner {
        self.owner
    }

    pub fn kind(&self) -> ObstacleKind {
        self.kind
    }

    pub fn priority(&self) -> ObstaclePriority {
        self.priority
    }

    /// Exclusive expiry tick.
    pub fn expires_at_tick(&self) -> u64 {
        self.expires_at_tick
    }

    pub fn created_at_tick(&self) -> u64 {
        self.created_at_tick
    }

    pub fn is_active(&self, now_tick: u64) -> bool {
        now_tick < self.expires_at_tick
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayInsert {
    Inserted { evicted: Option<EvictedObstacle> },
    Replaced,
    RejectedLowerPriority,
    RejectedOwnedByPeer,
    RejectedHardDowngrade,
    RejectedCapacity,
    RejectedInvalidTtl,
    RejectedZeroPenalty,
    RejectedClockRegression,
    RejectedRevisionExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictedObstacle {
    pub pos: BlockPos,
    pub owner: ObstacleOwner,
}

/// A frozen overlay generation and the first tick at which time alone can
/// change its effective entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayEpoch {
    pub revision: u64,
    pub valid_until_tick: Option<u64>,
}

impl OverlayEpoch {
    pub fn is_expired_at(self, now_tick: u64) -> bool {
        self.valid_until_tick
            .is_some_and(|expiry| now_tick >= expiry)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayTimeError {
    ClockRegressed { previous: u64, requested: u64 },
    RevisionExhausted,
}

impl fmt::Display for OverlayTimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClockRegressed {
                previous,
                requested,
            } => write!(
                f,
                "overlay clock regressed from tick {previous} to {requested}"
            ),
            Self::RevisionExhausted => write!(f, "overlay revision space is exhausted"),
        }
    }
}

impl std::error::Error for OverlayTimeError {}

/// A bounded obstacle set tied to exactly one world and dimension.
#[derive(Debug, Clone)]
pub struct DynamicObstacleOverlay {
    scope: WorldScope,
    capacity: usize,
    revision: u64,
    last_tick: u64,
    deadlock_policy: DeadlockPolicy,
    entries: HashMap<BlockPos, ObstacleEntry>,
}

impl DynamicObstacleOverlay {
    pub fn try_new(scope: WorldScope, capacity: usize) -> Result<Self, OverlayConfigError> {
        Self::try_new_with_policy(scope, capacity, DeadlockPolicy::default())
    }

    pub fn try_new_with_policy(
        scope: WorldScope,
        capacity: usize,
        deadlock_policy: DeadlockPolicy,
    ) -> Result<Self, OverlayConfigError> {
        if capacity == 0 || capacity > MAX_OVERLAY_ENTRIES {
            return Err(OverlayConfigError::InvalidCapacity {
                capacity,
                maximum: MAX_OVERLAY_ENTRIES,
            });
        }
        if deadlock_policy.max_hard_ticks == 0 || deadlock_policy.yielded_soft_penalty == 0 {
            return Err(OverlayConfigError::InvalidDeadlockPolicy);
        }
        Ok(Self {
            scope,
            capacity,
            revision: 0,
            last_tick: 0,
            deadlock_policy,
            entries: HashMap::with_capacity(capacity.min(1_024)),
        })
    }

    pub fn scope(&self) -> &WorldScope {
        &self.scope
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Monotonic generation for invalidating paths frozen against this overlay.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn last_tick(&self) -> u64 {
        self.last_tick
    }

    pub fn deadlock_policy(&self) -> DeadlockPolicy {
        self.deadlock_policy
    }

    /// Number of retained entries, including entries awaiting a prune call.
    pub fn retained_len(&self) -> usize {
        self.entries.len()
    }

    /// Number of entries active at the last successfully advanced tick.
    pub fn active_len(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.is_active(self.last_tick))
            .count()
    }

    /// Looks up an entry at the last successfully advanced tick.
    pub fn get(&self, pos: BlockPos) -> Option<&ObstacleEntry> {
        self.entries
            .get(&pos)
            .filter(|entry| entry.is_active(self.last_tick))
    }

    /// Removes an entry only when the caller proves ownership.
    pub fn remove_owned(
        &mut self,
        pos: BlockPos,
        owner: ObstacleOwner,
    ) -> Result<Option<ObstacleEntry>, OverlayTimeError> {
        if self
            .entries
            .get(&pos)
            .is_none_or(|entry| entry.owner != owner)
        {
            return Ok(None);
        }
        let next_revision = self.next_revision()?;
        let removed = self.entries.remove(&pos);
        self.revision = next_revision;
        Ok(removed)
    }

    /// Advances the overlay clock monotonically and makes TTL/deadlock
    /// transitions observable through [`Self::revision`].
    pub fn advance_to(&mut self, now_tick: u64) -> Result<usize, OverlayTimeError> {
        if now_tick < self.last_tick {
            return Err(OverlayTimeError::ClockRegressed {
                previous: self.last_tick,
                requested: now_tick,
            });
        }
        let crossed_hard_deadline = self.entries.values().any(|entry| {
            if entry.kind != ObstacleKind::Hard {
                return false;
            }
            let hard_until = entry
                .created_at_tick
                .saturating_add(self.deadlock_policy.max_hard_ticks)
                .min(entry.expires_at_tick);
            self.last_tick < hard_until && now_tick >= hard_until
        });
        let before = self.entries.len();
        let removed = self
            .entries
            .values()
            .filter(|entry| !entry.is_active(now_tick))
            .count();
        if removed > 0 || crossed_hard_deadline {
            let next_revision = self.next_revision()?;
            self.entries.retain(|_, entry| entry.is_active(now_tick));
            self.revision = next_revision;
        }
        debug_assert_eq!(before - removed, self.entries.len());
        self.last_tick = now_tick;
        Ok(removed)
    }

    fn next_revision(&self) -> Result<u64, OverlayTimeError> {
        self.revision
            .checked_add(1)
            .ok_or(OverlayTimeError::RevisionExhausted)
    }

    /// Inserts an obstacle with a TTL relative to `now_tick`.
    ///
    /// At capacity, the lowest-priority oldest entry is evicted only when the
    /// incoming observation is at least as important.  Thus noisy background
    /// observations cannot displace a critical collision report.
    pub fn insert(
        &mut self,
        pos: BlockPos,
        owner: ObstacleOwner,
        kind: ObstacleKind,
        priority: ObstaclePriority,
        now_tick: u64,
        ttl_ticks: u64,
    ) -> OverlayInsert {
        if ttl_ticks == 0 {
            return OverlayInsert::RejectedInvalidTtl;
        }
        let Some(expires_at_tick) = now_tick.checked_add(ttl_ticks) else {
            return OverlayInsert::RejectedInvalidTtl;
        };
        if matches!(kind, ObstacleKind::Soft { penalty: 0 }) {
            return OverlayInsert::RejectedZeroPenalty;
        }

        if let Err(error) = self.advance_to(now_tick) {
            return match error {
                OverlayTimeError::ClockRegressed { .. } => OverlayInsert::RejectedClockRegression,
                OverlayTimeError::RevisionExhausted => OverlayInsert::RejectedRevisionExhausted,
            };
        }
        if let Some(existing) = self.entries.get(&pos) {
            if priority < existing.priority {
                return OverlayInsert::RejectedLowerPriority;
            }
            if priority == existing.priority && owner != existing.owner {
                return OverlayInsert::RejectedOwnedByPeer;
            }
            if existing.kind == ObstacleKind::Hard && matches!(kind, ObstacleKind::Soft { .. }) {
                return OverlayInsert::RejectedHardDowngrade;
            }
            let Ok(next_revision) = self.next_revision() else {
                return OverlayInsert::RejectedRevisionExhausted;
            };
            self.entries.insert(
                pos,
                ObstacleEntry {
                    owner,
                    kind,
                    priority,
                    created_at_tick: now_tick,
                    expires_at_tick,
                    sequence: next_revision,
                },
            );
            self.revision = next_revision;
            return OverlayInsert::Replaced;
        }

        let Ok(next_revision) = self.next_revision() else {
            return OverlayInsert::RejectedRevisionExhausted;
        };
        let mut evicted = None;
        if self.entries.len() == self.capacity {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| (entry.priority, entry.sequence))
                .map(|(pos, entry)| (*pos, entry.owner, entry.priority));
            let Some((victim_pos, victim_owner, victim_priority)) = victim else {
                return OverlayInsert::RejectedCapacity;
            };
            if priority < victim_priority {
                return OverlayInsert::RejectedCapacity;
            }
            self.entries.remove(&victim_pos);
            evicted = Some(EvictedObstacle {
                pos: victim_pos,
                owner: victim_owner,
            });
        }

        self.entries.insert(
            pos,
            ObstacleEntry {
                owner,
                kind,
                priority,
                created_at_tick: now_tick,
                expires_at_tick,
                sequence: next_revision,
            },
        );
        self.revision = next_revision;
        OverlayInsert::Inserted { evicted }
    }

    fn active_entries(&self) -> impl Iterator<Item = (&BlockPos, &ObstacleEntry)> {
        self.entries
            .iter()
            .filter(move |(_, entry)| entry.is_active(self.last_tick))
    }

    /// Freezes the active overlay into a cloned move context and an overlay
    /// world view.  The input context is never mutated.
    ///
    /// Built-in movement rules consult the returned [`WorldView`], so hard
    /// entries are impassable.  Their maximum avoid toll is also copied into
    /// the context as a defensive signal for custom movement rules.  Custom
    /// rules must still honor the supplied world view to guarantee hard
    /// exclusion.
    pub fn apply<'a, W: WorldView + ?Sized>(
        &mut self,
        requested_scope: &WorldScope,
        now_tick: u64,
        viewer: Option<ObstacleOwner>,
        world: &'a W,
        input: &MoveContext,
    ) -> Result<AppliedOverlay<'a, W>, OverlayApplyError> {
        if requested_scope != &self.scope {
            return Err(OverlayApplyError::Scope(OverlayScopeMismatch {
                expected: self.scope.clone(),
                actual: requested_scope.clone(),
            }));
        }
        self.advance_to(now_tick).map_err(OverlayApplyError::Time)?;

        let mut avoid = (*input.avoid).clone();
        let mut hard = HashSet::new();
        let mut valid_until_tick = None;
        for (pos, entry) in self.active_entries() {
            if viewer == Some(entry.owner) {
                continue;
            }
            let hard_until = entry
                .created_at_tick
                .saturating_add(self.deadlock_policy.max_hard_ticks)
                .min(entry.expires_at_tick);
            let viewer_has_precedence = viewer.is_some_and(|viewer| viewer < entry.owner);
            let effective_kind = match entry.kind {
                ObstacleKind::Hard if now_tick >= hard_until || viewer_has_precedence => {
                    ObstacleKind::Soft {
                        penalty: self.deadlock_policy.yielded_soft_penalty,
                    }
                }
                kind => kind,
            };
            let next_change = if effective_kind == ObstacleKind::Hard {
                hard_until
            } else {
                entry.expires_at_tick
            };
            valid_until_tick =
                Some(valid_until_tick.map_or(next_change, |old: u64| old.min(next_change)));
            match effective_kind {
                ObstacleKind::Hard => {
                    hard.insert(*pos);
                    avoid.insert(*pos, Cost::MAX);
                }
                ObstacleKind::Soft { penalty } => {
                    avoid
                        .entry(*pos)
                        .and_modify(|existing| *existing = existing.saturating_add(penalty))
                        .or_insert(penalty);
                }
            }
        }
        let mut context = input.clone();
        context.avoid = Arc::new(avoid);
        Ok(AppliedOverlay {
            base: world,
            hard,
            context,
            overlay_epoch: OverlayEpoch {
                revision: self.revision,
                valid_until_tick,
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayConfigError {
    InvalidCapacity { capacity: usize, maximum: usize },
    InvalidDeadlockPolicy,
}

impl fmt::Display for OverlayConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity { capacity, maximum } => {
                write!(f, "overlay capacity is {capacity}; expected 1..={maximum}")
            }
            Self::InvalidDeadlockPolicy => write!(
                f,
                "deadlock policy requires a nonzero hard timeout and soft penalty"
            ),
        }
    }
}

impl std::error::Error for OverlayConfigError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayScopeMismatch {
    pub expected: WorldScope,
    pub actual: WorldScope,
}

impl fmt::Display for OverlayScopeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "overlay belongs to {}/{} but {}/{} was requested",
            self.expected.world(),
            self.expected.dimension(),
            self.actual.world(),
            self.actual.dimension()
        )
    }
}

impl std::error::Error for OverlayScopeMismatch {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayApplyError {
    Scope(OverlayScopeMismatch),
    Time(OverlayTimeError),
}

impl fmt::Display for OverlayApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scope(error) => error.fmt(f),
            Self::Time(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for OverlayApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scope(error) => Some(error),
            Self::Time(error) => Some(error),
        }
    }
}

/// A frozen context plus a hard-obstacle world view.
pub struct AppliedOverlay<'a, W: WorldView + ?Sized> {
    base: &'a W,
    hard: HashSet<BlockPos>,
    context: MoveContext,
    overlay_epoch: OverlayEpoch,
}

impl<W: WorldView + ?Sized> AppliedOverlay<'_, W> {
    pub fn context(&self) -> &MoveContext {
        &self.context
    }

    pub fn is_hard(&self, pos: BlockPos) -> bool {
        self.hard.contains(&pos)
    }

    pub fn hard_len(&self) -> usize {
        self.hard.len()
    }

    pub fn overlay_epoch(&self) -> OverlayEpoch {
        self.overlay_epoch
    }
}

impl<W: WorldView + ?Sized> WorldView for AppliedOverlay<'_, W> {
    fn block(&self, pos: BlockPos) -> BlockKind {
        if self.hard.contains(&pos) {
            // Unlike Solid this cannot become a floor for the block above.
            BlockKind::Fence
        } else {
            self.base.block(pos)
        }
    }

    fn standable(&self, pos: BlockPos) -> bool {
        // A hard body cell blocks both feet and head, and it must never become
        // artificial support for a landing one block above it.
        if self.hard.contains(&pos)
            || self.hard.contains(&offset(pos, 0, 1, 0))
            || self.hard.contains(&offset(pos, 0, 2, 0))
            || self.hard.contains(&offset(pos, 0, -1, 0))
        {
            return false;
        }
        self.base.standable(pos)
    }
}

/// Settings used to conservatively enumerate which blocks can affect an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DependencyOptions {
    /// Radius whose lava changes can alter the safety or cost of the edge.
    pub hazard_radius: i32,
    /// Local radius for collision, wall, water, and partial-block changes.
    pub local_neighborhood_radius: i32,
    /// Per-edge memory bound.  Exceeding it produces a broad dependency.
    pub max_blocks_per_edge: usize,
}

impl Default for DependencyOptions {
    fn default() -> Self {
        Self {
            hazard_radius: 4,
            local_neighborhood_radius: 1,
            max_blocks_per_edge: MAX_DEPENDENCIES_PER_EDGE,
        }
    }
}

impl DependencyOptions {
    pub fn from_context(ctx: &MoveContext) -> Self {
        let forbidden_clearance = match ctx.lava_policy {
            LavaPolicy::Forbidden { clearance } => clearance,
            LavaPolicy::Penalized => 0,
        };
        Self {
            hazard_radius: forbidden_clearance
                .max(ctx.lava_proximity_radius)
                .clamp(0, MAX_LAVA_SCAN_RADIUS),
            ..Self::default()
        }
    }

    fn sanitized(self) -> Self {
        Self {
            hazard_radius: self.hazard_radius.clamp(0, MAX_LAVA_SCAN_RADIUS),
            local_neighborhood_radius: self.local_neighborhood_radius.clamp(0, 2),
            max_blocks_per_edge: self.max_blocks_per_edge.clamp(1, MAX_DEPENDENCIES_PER_EDGE),
        }
    }
}

/// Blocks whose state may affect a single transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyFootprint {
    pub edge_index: usize,
    pub from: BlockPos,
    pub to: BlockPos,
    blocks: Vec<BlockPos>,
    invalidate_on_any_change: bool,
}

impl DependencyFootprint {
    /// Creates a bounded exact footprint supplied by a motion generator.
    ///
    /// More than [`MAX_DEPENDENCIES_PER_EDGE`] unique blocks degrades to a
    /// broad footprint rather than allocating without a bound.
    pub fn from_generator_blocks(
        edge_index: usize,
        from: BlockPos,
        to: BlockPos,
        blocks: impl IntoIterator<Item = BlockPos>,
    ) -> Self {
        let mut builder = FootprintBuilder::new(MAX_DEPENDENCIES_PER_EDGE);
        for block in blocks {
            builder.add(block);
            if builder.broad {
                break;
            }
        }
        if builder.blocks.is_empty() {
            builder.broad = true;
        }
        builder.finish(edge_index, from, to)
    }

    /// Creates an explicitly broad footprint for an unknown/custom primitive.
    pub fn broad(edge_index: usize, from: BlockPos, to: BlockPos) -> Self {
        Self {
            edge_index,
            from,
            to,
            blocks: Vec::new(),
            invalidate_on_any_change: true,
        }
    }

    pub fn blocks(&self) -> &[BlockPos] {
        &self.blocks
    }

    pub fn contains(&self, pos: BlockPos) -> bool {
        self.blocks
            .binary_search_by_key(&(pos.x, pos.y, pos.z), |block| (block.x, block.y, block.z))
            .is_ok()
    }

    /// True when the exact conservative set exceeded its allocation bound.
    pub fn invalidates_on_any_change(&self) -> bool {
        self.invalidate_on_any_change
    }
}

struct FootprintBuilder {
    blocks: HashSet<BlockPos>,
    maximum: usize,
    broad: bool,
}

impl FootprintBuilder {
    fn new(maximum: usize) -> Self {
        Self {
            blocks: HashSet::with_capacity(maximum.min(1_024)),
            maximum,
            broad: false,
        }
    }

    fn add(&mut self, pos: BlockPos) {
        if self.broad || self.blocks.contains(&pos) {
            return;
        }
        if self.blocks.len() >= self.maximum {
            self.broad = true;
            return;
        }
        self.blocks.insert(pos);
    }

    fn body(&mut self, pos: BlockPos) {
        if self.broad {
            return;
        }
        self.add(offset(pos, 0, -1, 0)); // support / fluid below
        self.add(pos); // feet
        self.add(offset(pos, 0, 1, 0)); // head
        self.add(offset(pos, 0, 2, 0)); // jump/climb headroom
    }

    fn neighborhood(&mut self, pos: BlockPos, radius: i32) {
        if self.broad {
            return;
        }
        for dx in -radius..=radius {
            for dz in -radius..=radius {
                for dy in -1..=2 {
                    self.add(offset(pos, dx, dy, dz));
                    if self.broad {
                        return;
                    }
                }
            }
        }
    }

    fn finish(mut self, edge_index: usize, from: BlockPos, to: BlockPos) -> DependencyFootprint {
        let mut blocks: Vec<_> = self.blocks.drain().collect();
        blocks.sort_unstable_by_key(|block| (block.x, block.y, block.z));
        DependencyFootprint {
            edge_index,
            from,
            to,
            blocks,
            invalidate_on_any_change: self.broad,
        }
    }
}

fn lerp_component(from: i32, delta: i64, step: i64, steps: i64) -> i32 {
    if steps == 0 {
        return from;
    }
    let value = i64::from(from).saturating_add(delta.saturating_mul(step) / steps);
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn transition_samples(
    from: BlockPos,
    to: BlockPos,
    builder: &mut FootprintBuilder,
) -> Vec<BlockPos> {
    let dx = i64::from(to.x) - i64::from(from.x);
    let dy = i64::from(to.y) - i64::from(from.y);
    let dz = i64::from(to.z) - i64::from(from.z);
    let steps = dx.abs().max(dy.abs()).max(dz.abs());
    if steps > MAX_SAMPLED_TRANSITION_LENGTH {
        // An unmodeled long-range primitive cannot safely claim a narrow
        // dependency set.  Keep endpoint information for diagnostics and make
        // the edge respond to every update.
        builder.broad = true;
        return vec![from, to];
    }
    if steps == 0 {
        return vec![from];
    }
    (0..=steps)
        .map(|step| {
            BlockPos::new(
                lerp_component(from.x, dx, step, steps),
                lerp_component(from.y, dy, step, steps),
                lerp_component(from.z, dz, step, steps),
            )
        })
        .collect()
}

/// Computes a bounded conservative footprint for one path transition.
///
/// The footprint contains both endpoint bodies and supports, diagonal corner
/// clearance, jump/parkour headroom, the landing neighborhood, climb
/// attachments, the swim/collision neighborhood, and the configured lava
/// safety/cost radius.  A caller may therefore invalidate a route without
/// inspecting the new block kind first.
pub fn transition_dependency_footprint(
    edge_index: usize,
    from: BlockPos,
    to: BlockPos,
    kind: MoveKind,
    options: DependencyOptions,
) -> DependencyFootprint {
    let options = options.sanitized();
    let mut builder = FootprintBuilder::new(options.max_blocks_per_edge);
    macro_rules! finish_if_broad {
        () => {
            if builder.broad {
                return builder.finish(edge_index, from, to);
            }
        };
    }
    builder.body(from);
    builder.body(to);
    if !known_builtin_geometry(from, to, kind) {
        builder.broad = true;
    }
    finish_if_broad!();

    let samples = transition_samples(from, to, &mut builder);
    finish_if_broad!();
    for sample in &samples {
        builder.body(*sample);
        builder.neighborhood(*sample, options.local_neighborhood_radius);
        finish_if_broad!();
    }

    let dx = to.x.saturating_sub(from.x);
    let dz = to.z.saturating_sub(from.z);
    if dx != 0 && dz != 0 {
        // Both orthogonal body columns must remain clear for a diagonal.
        for corner in [
            BlockPos::new(to.x, from.y, from.z),
            BlockPos::new(from.x, from.y, to.z),
        ] {
            builder.body(corner);
            finish_if_broad!();
        }
    }

    if matches!(kind, MoveKind::Jump | MoveKind::Parkour { .. }) {
        // The precise continuous arc is executor-dependent.  Two extra blocks
        // above every discrete corridor sample is a safe Minecraft-sized
        // clearance envelope for the built-in jump and parkour primitives.
        for sample in &samples {
            builder.body(offset(*sample, 0, 1, 0));
            builder.body(offset(*sample, 0, 2, 0));
            finish_if_broad!();
        }
    }

    if matches!(kind, MoveKind::Parkour { .. }) {
        let span_x = i64::from(to.x).abs_diff(i64::from(from.x));
        let span_z = i64::from(to.z).abs_diff(i64::from(from.z));
        let span_y = i64::from(to.y).abs_diff(i64::from(from.y));
        if span_x.max(span_y).max(span_z) > MAX_SAMPLED_TRANSITION_LENGTH as u64 {
            builder.broad = true;
        } else {
            // Parkour validation samples the crossed horizontal columns more
            // finely than a voxel DDA and, for descending jumps, checks their
            // full vertical corridor.  The small built-in reach (four blocks)
            // makes this complete bounding prism both safer and cheaper than
            // trying to reproduce floating-point samples here.
            let low_y = from.y.min(to.y).saturating_sub(1);
            let high_y = from.y.max(to.y).saturating_add(3);
            for x in from.x.min(to.x)..=from.x.max(to.x) {
                for z in from.z.min(to.z)..=from.z.max(to.z) {
                    for y in low_y..=high_y {
                        builder.add(BlockPos::new(x, y, z));
                        finish_if_broad!();
                    }
                }
            }

            // Long and rising jumps also depend on one or two blocks of
            // run-up.  Diagonal jumps accept the diagonal approach or either
            // cardinal half, across one block of vertical variation.
            let sx = to.x.saturating_sub(from.x).signum();
            let sz = to.z.saturating_sub(from.z).signum();
            for back in 1..=2 {
                for (bx, bz) in [
                    (-sx.saturating_mul(back), -sz.saturating_mul(back)),
                    (-sx.saturating_mul(back), 0),
                    (0, -sz.saturating_mul(back)),
                ] {
                    if (bx, bz) == (0, 0) {
                        continue;
                    }
                    for dy in -1..=1 {
                        builder.body(offset(from, bx, dy, bz));
                        finish_if_broad!();
                    }
                }
            }
        }
    }

    if kind == MoveKind::Fall {
        let span_y = i64::from(to.y).abs_diff(i64::from(from.y));
        if span_y > MAX_SAMPLED_TRANSITION_LENGTH as u64 {
            builder.broad = true;
        } else {
            // FallMove steps sideways first and then descends in the landing
            // column.  A straight 3-D DDA can remain in the takeoff column for
            // most of a steep fall, so record the actual descent explicitly.
            for y in from.y.min(to.y)..=from.y.max(to.y) {
                builder.body(BlockPos::new(to.x, y, to.z));
                finish_if_broad!();
            }
        }
    }

    if kind == MoveKind::Climb {
        // A ladder/vine/scaffold and any cardinal attachment/support may
        // determine whether the whole vertical transition is still possible.
        for sample in &samples {
            builder.body(*sample);
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                builder.body(offset(*sample, dx, 0, dz));
                finish_if_broad!();
            }
        }
    }

    if kind == MoveKind::Swim {
        // Water can disappear or appear beside the body between snapshots.
        for sample in &samples {
            builder.neighborhood(*sample, 1);
            finish_if_broad!();
        }
    }

    // Landing/support geometry can interact with the player's 0.6-block-wide
    // hitbox, so include the complete 3x3 landing column neighborhood.
    builder.neighborhood(to, 1);
    finish_if_broad!();

    if options.hazard_radius > 0 {
        for sample in &samples {
            builder.neighborhood(*sample, options.hazard_radius);
            finish_if_broad!();
        }
    }

    builder.finish(edge_index, from, to)
}

fn known_builtin_geometry(from: BlockPos, to: BlockPos, kind: MoveKind) -> bool {
    let dx = i64::from(to.x) - i64::from(from.x);
    let dy = i64::from(to.y) - i64::from(from.y);
    let dz = i64::from(to.z) - i64::from(from.z);
    let ax = dx.unsigned_abs();
    let ay = dy.unsigned_abs();
    let az = dz.unsigned_abs();
    let horizontal_axes = u8::from(ax > 0) + u8::from(az > 0);
    match kind {
        MoveKind::Start | MoveKind::Aotv | MoveKind::Etherwarp => false,
        MoveKind::Walk => {
            ax <= 1
                && az <= 1
                && horizontal_axes > 0
                && ((horizontal_axes == 1 && ay <= 1) || (horizontal_axes == 2 && dy == 0))
        }
        MoveKind::Jump => horizontal_axes == 1 && ax.max(az) == 1 && dy == 1,
        MoveKind::Fall => horizontal_axes == 1 && ax.max(az) == 1 && dy < 0,
        MoveKind::Parkour { blocks, rise } => {
            let squared = dx.saturating_mul(dx).saturating_add(dz.saturating_mul(dz));
            let distance = (squared as f64).sqrt();
            (2.0..=4.0).contains(&distance)
                && i64::from(rise) == dy
                && u8::try_from(distance.round() as i64) == Ok(blocks)
        }
        MoveKind::Climb => {
            ax <= 1
                && az <= 1
                && ay <= 1
                && ((horizontal_axes == 0 && ay == 1)
                    || (horizontal_axes == 1 && (dy == 0 || dy == 1)))
        }
        MoveKind::Swim => ax <= 1 && az <= 1 && ay <= 1 && horizontal_axes <= 1 && ax + ay + az > 0,
    }
}

/// Frozen revisions against which a path and its dependency index were built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteRevisions {
    pub path: u64,
    pub world_basis: u64,
    pub cost_model: u64,
    pub dynamic_obstacles: OverlayEpoch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkColumn {
    pub x: i32,
    pub z: i32,
}

impl ChunkColumn {
    pub fn containing(pos: BlockPos) -> Self {
        Self {
            x: pos.x.div_euclid(16),
            z: pos.z.div_euclid(16),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkChangeKind {
    Loaded,
    Unloaded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkChange {
    pub column: ChunkColumn,
    pub kind: ChunkChangeKind,
}

/// A complete change interval since the world basis used for planning.
pub struct WorldChangeBatch<'a> {
    pub scope: &'a WorldScope,
    pub path_revision: u64,
    pub from_world_revision: u64,
    pub to_world_revision: u64,
    pub current_cost_model: u64,
    pub current_dynamic_obstacles: OverlayEpoch,
    pub now_tick: u64,
    pub changed_blocks: &'a [BlockPos],
    pub changed_chunks: &'a [ChunkChange],
}

/// Reverse lookup from changed block to the earliest path edge it may affect.
#[derive(Debug, Clone)]
pub struct PathDependencyIndex {
    scope: WorldScope,
    revisions: RouteRevisions,
    edge_count: usize,
    earliest_by_block: HashMap<BlockPos, usize>,
    earliest_by_chunk: HashMap<ChunkColumn, usize>,
    broad_from_edge: Option<usize>,
}

impl PathDependencyIndex {
    /// Builds inferred footprints for paths produced by this crate's built-in
    /// movement registry.  Teleports and geometries outside built-in bounds
    /// become broad dependencies.
    pub fn build_builtin(
        path: &Path,
        scope: WorldScope,
        revisions: RouteRevisions,
        ctx: &MoveContext,
    ) -> Self {
        let options = DependencyOptions::from_context(ctx);
        let mut index = Self {
            scope,
            revisions,
            edge_count: path.nodes.len().saturating_sub(1),
            earliest_by_block: HashMap::new(),
            earliest_by_chunk: HashMap::new(),
            broad_from_edge: None,
        };

        for (edge_index, nodes) in path.nodes.windows(2).enumerate() {
            let footprint = transition_dependency_footprint(
                edge_index,
                nodes[0].pos,
                nodes[1].pos,
                nodes[1].reached_by,
                options,
            );
            index.add_footprint(&footprint);
            if index
                .broad_from_edge
                .is_some_and(|broad| broad <= edge_index)
            {
                break;
            }
        }
        index
    }

    /// Safest fallback for a custom or otherwise untrusted motion registry.
    ///
    /// Without generator-authored dependencies, every world change may affect
    /// the first edge.  This deliberately chooses extra replans over a false
    /// claim that a custom transition is independent of distant state.
    pub fn build_unknown(path: &Path, scope: WorldScope, revisions: RouteRevisions) -> Self {
        let edge_count = path.nodes.len().saturating_sub(1);
        Self {
            scope,
            revisions,
            edge_count,
            earliest_by_block: HashMap::new(),
            earliest_by_chunk: HashMap::new(),
            broad_from_edge: (edge_count > 0).then_some(0),
        }
    }

    /// Constructs an index from precomputed footprints.
    ///
    /// This is useful when a motion primitive supplies a footprint more exact
    /// than the built-in conservative geometry.
    pub fn from_footprints(
        path: &Path,
        scope: WorldScope,
        revisions: RouteRevisions,
        footprints: &[DependencyFootprint],
    ) -> Result<Self, DependencyIndexError> {
        let edge_count = path.nodes.len().saturating_sub(1);
        let mut index = Self {
            scope,
            revisions,
            edge_count,
            earliest_by_block: HashMap::new(),
            earliest_by_chunk: HashMap::new(),
            broad_from_edge: None,
        };
        let mut by_edge: Vec<Option<&DependencyFootprint>> = vec![None; edge_count];
        for footprint in footprints {
            if footprint.edge_index >= edge_count {
                return Err(DependencyIndexError::EdgeOutOfRange {
                    edge: footprint.edge_index,
                    edge_count,
                });
            }
            if by_edge[footprint.edge_index].replace(footprint).is_some() {
                index.broad_from_edge = Some(
                    index
                        .broad_from_edge
                        .map_or(footprint.edge_index, |old| old.min(footprint.edge_index)),
                );
            }
        }
        for (edge_index, nodes) in path.nodes.windows(2).enumerate() {
            let Some(footprint) = by_edge[edge_index] else {
                index.broad_from_edge = Some(
                    index
                        .broad_from_edge
                        .map_or(edge_index, |old| old.min(edge_index)),
                );
                break;
            };
            if footprint.from != nodes[0].pos || footprint.to != nodes[1].pos {
                index.broad_from_edge = Some(
                    index
                        .broad_from_edge
                        .map_or(edge_index, |old| old.min(edge_index)),
                );
                break;
            }
            index.add_footprint(footprint);
            if index
                .broad_from_edge
                .is_some_and(|broad| broad <= edge_index)
            {
                break;
            }
        }
        Ok(index)
    }

    fn add_footprint(&mut self, footprint: &DependencyFootprint) {
        if footprint.invalidate_on_any_change || footprint.blocks.is_empty() {
            self.broad_from_edge = Some(
                self.broad_from_edge
                    .map_or(footprint.edge_index, |old| old.min(footprint.edge_index)),
            );
            return;
        }
        for pos in &footprint.blocks {
            if self.earliest_by_block.len() >= MAX_INDEXED_DEPENDENCIES
                && !self.earliest_by_block.contains_key(pos)
            {
                self.broad_from_edge = Some(
                    self.broad_from_edge
                        .map_or(footprint.edge_index, |old| old.min(footprint.edge_index)),
                );
                break;
            }
            self.earliest_by_block
                .entry(*pos)
                .and_modify(|edge| *edge = (*edge).min(footprint.edge_index))
                .or_insert(footprint.edge_index);
            self.earliest_by_chunk
                .entry(ChunkColumn::containing(*pos))
                .and_modify(|edge| *edge = (*edge).min(footprint.edge_index))
                .or_insert(footprint.edge_index);
        }
    }

    pub fn scope(&self) -> &WorldScope {
        &self.scope
    }

    pub fn revisions(&self) -> RouteRevisions {
        self.revisions
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn indexed_blocks(&self) -> usize {
        self.earliest_by_block.len()
    }

    /// Returns the earliest edge touched by a complete change interval, or
    /// `None` when the interval is unrelated.
    ///
    /// The batch must identify the exact scope and frozen path/world/cost
    /// revisions used to build this index.  A mismatch is rejected rather than
    /// accidentally validating a route against a different planning epoch.
    pub fn earliest_affected(
        &mut self,
        batch: &WorldChangeBatch<'_>,
    ) -> Result<Option<usize>, InvalidationError> {
        if batch.scope != &self.scope {
            return Err(InvalidationError::ScopeMismatch {
                expected: self.scope.clone(),
                actual: batch.scope.clone(),
            });
        }
        if batch.path_revision != self.revisions.path {
            return Err(InvalidationError::PathRevisionMismatch {
                expected: self.revisions.path,
                actual: batch.path_revision,
            });
        }
        if batch.from_world_revision != self.revisions.world_basis {
            return Err(InvalidationError::WorldContinuityMismatch {
                expected_from: self.revisions.world_basis,
                actual_from: batch.from_world_revision,
            });
        }
        if batch.to_world_revision < batch.from_world_revision {
            return Err(InvalidationError::WorldRevisionRegressed {
                from: batch.from_world_revision,
                to: batch.to_world_revision,
            });
        }
        if self.edge_count == 0 {
            self.revisions.world_basis = batch.to_world_revision;
            return Ok(None);
        }

        // A cost or obstacle epoch change invalidates the complete route, not
        // the integrity of the change batch.  Treat it as an earliest-edge
        // hit rather than conflating it with malformed revision metadata.
        if batch.current_cost_model != self.revisions.cost_model
            || batch.current_dynamic_obstacles != self.revisions.dynamic_obstacles
            || self
                .revisions
                .dynamic_obstacles
                .is_expired_at(batch.now_tick)
        {
            return Ok(Some(0));
        }

        let world_changed = batch.to_world_revision > batch.from_world_revision;
        if !world_changed && (!batch.changed_blocks.is_empty() || !batch.changed_chunks.is_empty())
        {
            return Err(InvalidationError::WorldChangeWithoutRevision {
                revision: batch.from_world_revision,
            });
        }
        if world_changed && batch.changed_blocks.is_empty() && batch.changed_chunks.is_empty() {
            // A revision advanced without details.  Failing broad is the only
            // sound choice; it may represent a bulk chunk replacement.
            return Ok(Some(0));
        }
        if !world_changed && batch.changed_blocks.is_empty() && batch.changed_chunks.is_empty() {
            return Ok(None);
        }

        let mut earliest = self.broad_from_edge;
        for changed in batch.changed_blocks {
            if let Some(edge) = self.earliest_by_block.get(changed) {
                earliest = Some(earliest.map_or(*edge, |old| old.min(*edge)));
            }
            if earliest == Some(0) {
                return Ok(earliest);
            }
        }
        for changed in batch.changed_chunks {
            if let Some(edge) = self.earliest_by_chunk.get(&changed.column) {
                earliest = Some(earliest.map_or(*edge, |old| old.min(*edge)));
            }
            if earliest == Some(0) {
                return Ok(earliest);
            }
        }
        if earliest.is_none() {
            self.revisions.world_basis = batch.to_world_revision;
        }
        Ok(earliest)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyIndexError {
    EdgeOutOfRange { edge: usize, edge_count: usize },
}

impl fmt::Display for DependencyIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EdgeOutOfRange { edge, edge_count } => {
                write!(
                    f,
                    "dependency edge {edge} is outside {edge_count} path edges"
                )
            }
        }
    }
}

impl std::error::Error for DependencyIndexError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidationError {
    ScopeMismatch {
        expected: WorldScope,
        actual: WorldScope,
    },
    PathRevisionMismatch {
        expected: u64,
        actual: u64,
    },
    WorldContinuityMismatch {
        expected_from: u64,
        actual_from: u64,
    },
    WorldRevisionRegressed {
        from: u64,
        to: u64,
    },
    WorldChangeWithoutRevision {
        revision: u64,
    },
}

impl fmt::Display for InvalidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScopeMismatch { expected, actual } => write!(
                f,
                "path scope is {}/{} but update scope is {}/{}",
                expected.world(),
                expected.dimension(),
                actual.world(),
                actual.dimension()
            ),
            Self::PathRevisionMismatch { expected, actual } => {
                write!(f, "path revision is {expected} but update uses {actual}")
            }
            Self::WorldContinuityMismatch {
                expected_from,
                actual_from,
            } => write!(
                f,
                "world change interval starts at {actual_from}; path requires {expected_from}"
            ),
            Self::WorldRevisionRegressed { from, to } => {
                write!(f, "world change interval regresses from {from} to {to}")
            }
            Self::WorldChangeWithoutRevision { revision } => write!(
                f,
                "world changes were supplied without advancing revision {revision}"
            ),
        }
    }
}

impl std::error::Error for InvalidationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSegment {
    Original,
    Patch,
    Joined(usize),
}

pub const MAX_STRUCTURAL_PATH_NODES: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSpliceError {
    NoSegments,
    Empty {
        segment: PathSegment,
    },
    InvalidStartMarker {
        segment: PathSegment,
    },
    EmbeddedStart {
        segment: PathSegment,
        node: usize,
    },
    InvalidRange {
        start_node: usize,
        end_node: usize,
        node_count: usize,
    },
    PatchHasNoEdge,
    EndpointMismatch {
        segment: PathSegment,
        expected: BlockPos,
        actual: BlockPos,
    },
    TooManyNodes {
        nodes: usize,
        maximum: usize,
    },
    AllocationFailed,
    ValidatorScopeMismatch {
        expected: WorldScope,
        actual: WorldScope,
    },
    ValidatorRevisionMismatch {
        expected: RouteRevisions,
        actual: RouteRevisions,
    },
    ValidatorEpochExpired {
        valid_until_tick: u64,
        validated_at_tick: u64,
    },
    EdgeRejected {
        edge: usize,
        reason: EdgeValidationError,
    },
    CostOverflow,
}

impl fmt::Display for PathSpliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSegments => write!(f, "no complete paths were supplied"),
            Self::Empty { segment } => write!(f, "{segment:?} path is empty"),
            Self::InvalidStartMarker { segment } => {
                write!(f, "{segment:?} path does not begin with MoveKind::Start")
            }
            Self::EmbeddedStart { segment, node } => {
                write!(
                    f,
                    "{segment:?} path has an embedded start marker at node {node}"
                )
            }
            Self::InvalidRange {
                start_node,
                end_node,
                node_count,
            } => write!(
                f,
                "splice range {start_node}..={end_node} is invalid for {node_count} nodes"
            ),
            Self::PatchHasNoEdge => write!(f, "patch must contain at least one edge"),
            Self::EndpointMismatch {
                segment,
                expected,
                actual,
            } => write!(
                f,
                "{segment:?} path endpoint mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::TooManyNodes { nodes, maximum } => {
                write!(f, "structural path has {nodes} nodes; maximum is {maximum}")
            }
            Self::AllocationFailed => write!(f, "could not allocate structural path"),
            Self::ValidatorScopeMismatch { expected, actual } => write!(
                f,
                "validator scope is {}/{}; expected {}/{}",
                actual.world(),
                actual.dimension(),
                expected.world(),
                expected.dimension()
            ),
            Self::ValidatorRevisionMismatch { expected, actual } => write!(
                f,
                "validator revisions are {actual:?}; expected {expected:?}"
            ),
            Self::ValidatorEpochExpired {
                valid_until_tick,
                validated_at_tick,
            } => write!(
                f,
                "overlay epoch expired at tick {valid_until_tick} before validation tick {validated_at_tick}"
            ),
            Self::EdgeRejected { edge, reason } => {
                write!(f, "validator rejected edge {edge}: {reason:?}")
            }
            Self::CostOverflow => write!(f, "validated path cost overflowed"),
        }
    }
}

impl std::error::Error for PathSpliceError {}

fn validate_complete_path(path: &Path, segment: PathSegment) -> Result<(), PathSpliceError> {
    let Some(first) = path.nodes.first() else {
        return Err(PathSpliceError::Empty { segment });
    };
    if first.reached_by != MoveKind::Start {
        return Err(PathSpliceError::InvalidStartMarker { segment });
    }
    if let Some((node, _)) = path
        .nodes
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, node)| node.reached_by == MoveKind::Start)
    {
        return Err(PathSpliceError::EmbeddedStart { segment, node });
    }
    Ok(())
}

/// Node structure without executable validity or cost claims.
///
/// There is intentionally no conversion to [`Path`] without an
/// [`PathEdgeValidator`].  Endpoint equality alone cannot prove that a seam or
/// a replacement edge remains legal in the current world.
#[derive(Debug, Clone)]
pub struct UnvalidatedStructuralPath {
    nodes: Vec<PathNode>,
}

impl UnvalidatedStructuralPath {
    pub fn nodes(&self) -> &[PathNode] {
        &self.nodes
    }
}

/// Structurally joins complete paths at matching endpoint positions.
///
/// The result is not executable until [`validate_structural_path`] validates
/// every edge against a frozen scope and revision set.
pub fn structurally_join_paths(
    paths: &[&Path],
) -> Result<UnvalidatedStructuralPath, PathSpliceError> {
    let Some(first) = paths.first() else {
        return Err(PathSpliceError::NoSegments);
    };
    validate_complete_path(first, PathSegment::Joined(0))?;
    let requested = paths.iter().try_fold(0usize, |total, path| {
        total.checked_add(path.nodes.len().saturating_sub(1))
    });
    let Some(requested) = requested.and_then(|edges| edges.checked_add(1)) else {
        return Err(PathSpliceError::TooManyNodes {
            nodes: usize::MAX,
            maximum: MAX_STRUCTURAL_PATH_NODES,
        });
    };
    if requested > MAX_STRUCTURAL_PATH_NODES {
        return Err(PathSpliceError::TooManyNodes {
            nodes: requested,
            maximum: MAX_STRUCTURAL_PATH_NODES,
        });
    }
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(requested)
        .map_err(|_| PathSpliceError::AllocationFailed)?;
    nodes.extend_from_slice(&first.nodes);
    for (index, path) in paths.iter().enumerate().skip(1) {
        let segment = PathSegment::Joined(index);
        validate_complete_path(path, segment)?;
        let expected = nodes.last().expect("validated nonempty path").pos;
        let actual = path.nodes.first().expect("validated nonempty path").pos;
        if expected != actual {
            return Err(PathSpliceError::EndpointMismatch {
                segment,
                expected,
                actual,
            });
        }
        nodes.extend_from_slice(&path.nodes[1..]);
    }
    Ok(UnvalidatedStructuralPath { nodes })
}

/// Structurally replaces an inclusive node range with a complete patch.
///
/// This only checks node structure and matching endpoints.  The returned value
/// is deliberately not a [`Path`] and must be passed through
/// [`validate_structural_path`] before execution.
pub fn structurally_splice_path(
    original: &Path,
    start_node: usize,
    end_node: usize,
    patch: &Path,
) -> Result<UnvalidatedStructuralPath, PathSpliceError> {
    validate_complete_path(original, PathSegment::Original)?;
    validate_complete_path(patch, PathSegment::Patch)?;
    if start_node >= end_node || end_node >= original.nodes.len() {
        return Err(PathSpliceError::InvalidRange {
            start_node,
            end_node,
            node_count: original.nodes.len(),
        });
    }
    if patch.nodes.len() < 2 {
        return Err(PathSpliceError::PatchHasNoEdge);
    }
    let expected_start = original.nodes[start_node].pos;
    let actual_start = patch.nodes[0].pos;
    if expected_start != actual_start {
        return Err(PathSpliceError::EndpointMismatch {
            segment: PathSegment::Patch,
            expected: expected_start,
            actual: actual_start,
        });
    }
    let expected_end = original.nodes[end_node].pos;
    let actual_end = patch
        .nodes
        .last()
        .expect("patch has at least two nodes")
        .pos;
    if expected_end != actual_end {
        return Err(PathSpliceError::EndpointMismatch {
            segment: PathSegment::Patch,
            expected: expected_end,
            actual: actual_end,
        });
    }

    let Some(node_count) = original
        .nodes
        .len()
        .checked_sub(end_node - start_node)
        .and_then(|count| count.checked_add(patch.nodes.len().saturating_sub(1)))
    else {
        return Err(PathSpliceError::TooManyNodes {
            nodes: usize::MAX,
            maximum: MAX_STRUCTURAL_PATH_NODES,
        });
    };
    if node_count > MAX_STRUCTURAL_PATH_NODES {
        return Err(PathSpliceError::TooManyNodes {
            nodes: node_count,
            maximum: MAX_STRUCTURAL_PATH_NODES,
        });
    }
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(node_count)
        .map_err(|_| PathSpliceError::AllocationFailed)?;
    nodes.extend_from_slice(&original.nodes[..=start_node]);
    nodes.extend_from_slice(&patch.nodes[1..]);
    nodes.extend_from_slice(&original.nodes[end_node + 1..]);
    Ok(UnvalidatedStructuralPath { nodes })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeValidationError {
    Unsafe,
    UnknownPrimitive,
    StaleWorld,
    CostUnavailable,
}

/// Live validator used to turn structural nodes into an executable route.
pub trait PathEdgeValidator {
    fn scope(&self) -> &WorldScope;
    fn revisions(&self) -> RouteRevisions;
    fn validation_tick(&self) -> u64;
    fn validate_edge(
        &mut self,
        from: BlockPos,
        to: BlockPos,
        reached_by: MoveKind,
    ) -> Result<Cost, EdgeValidationError>;
}

/// Executable path tied to the exact scope and revisions used for validation.
#[derive(Debug, Clone)]
pub struct ValidatedPath {
    path: Path,
    scope: WorldScope,
    revisions: RouteRevisions,
    validated_at_tick: u64,
}

impl ValidatedPath {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn scope(&self) -> &WorldScope {
        &self.scope
    }

    pub fn revisions(&self) -> RouteRevisions {
        self.revisions
    }

    pub fn validated_at_tick(&self) -> u64 {
        self.validated_at_tick
    }
}

/// Validates every edge and regenerates the exact aggregate cost.
pub fn validate_structural_path<V: PathEdgeValidator + ?Sized>(
    structural: UnvalidatedStructuralPath,
    expected_scope: &WorldScope,
    expected_revisions: RouteRevisions,
    validator: &mut V,
) -> Result<ValidatedPath, PathSpliceError> {
    if validator.scope() != expected_scope {
        return Err(PathSpliceError::ValidatorScopeMismatch {
            expected: expected_scope.clone(),
            actual: validator.scope().clone(),
        });
    }
    if validator.revisions() != expected_revisions {
        return Err(PathSpliceError::ValidatorRevisionMismatch {
            expected: expected_revisions,
            actual: validator.revisions(),
        });
    }
    let validated_at_tick = validator.validation_tick();
    if let Some(valid_until_tick) = expected_revisions.dynamic_obstacles.valid_until_tick
        && validated_at_tick >= valid_until_tick
    {
        return Err(PathSpliceError::ValidatorEpochExpired {
            valid_until_tick,
            validated_at_tick,
        });
    }
    let mut total_cost: Cost = 0;
    for (edge, nodes) in structural.nodes.windows(2).enumerate() {
        let edge_cost = validator
            .validate_edge(nodes[0].pos, nodes[1].pos, nodes[1].reached_by)
            .map_err(|reason| PathSpliceError::EdgeRejected { edge, reason })?;
        total_cost = total_cost
            .checked_add(edge_cost)
            .ok_or(PathSpliceError::CostOverflow)?;
    }
    Ok(ValidatedPath {
        path: Path {
            nodes: structural.nodes,
            total_cost,
        },
        scope: expected_scope.clone(),
        revisions: expected_revisions,
        validated_at_tick,
    })
}

pub fn splice_path_validated<V: PathEdgeValidator + ?Sized>(
    original: &Path,
    start_node: usize,
    end_node: usize,
    patch: &Path,
    expected_scope: &WorldScope,
    expected_revisions: RouteRevisions,
    validator: &mut V,
) -> Result<ValidatedPath, PathSpliceError> {
    let structural = structurally_splice_path(original, start_node, end_node, patch)?;
    validate_structural_path(structural, expected_scope, expected_revisions, validator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(world: &str, dimension: &str) -> WorldScope {
        WorldScope::try_new(world, dimension).unwrap()
    }

    fn pos(x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos::new(x, y, z)
    }

    fn node(x: i32, kind: MoveKind) -> PathNode {
        PathNode {
            pos: pos(x, 64, 0),
            reached_by: kind,
        }
    }

    fn path(xs: &[i32], cost: Cost) -> Path {
        Path {
            nodes: xs
                .iter()
                .enumerate()
                .map(|(index, x)| {
                    node(
                        *x,
                        if index == 0 {
                            MoveKind::Start
                        } else {
                            MoveKind::Walk
                        },
                    )
                })
                .collect(),
            total_cost: cost,
        }
    }

    const OWNER: ObstacleOwner = ObstacleOwner(10);

    fn epoch(revision: u64) -> OverlayEpoch {
        OverlayEpoch {
            revision,
            valid_until_tick: None,
        }
    }

    fn revisions(path: u64, world: u64, cost: u64, overlay: u64) -> RouteRevisions {
        RouteRevisions {
            path,
            world_basis: world,
            cost_model: cost,
            dynamic_obstacles: epoch(overlay),
        }
    }

    fn batch<'a>(
        key: &'a WorldScope,
        revisions: RouteRevisions,
        blocks: &'a [BlockPos],
        chunks: &'a [ChunkChange],
    ) -> WorldChangeBatch<'a> {
        WorldChangeBatch {
            scope: key,
            path_revision: revisions.path,
            from_world_revision: revisions.world_basis,
            to_world_revision: revisions
                .world_basis
                .saturating_add(u64::from(!blocks.is_empty() || !chunks.is_empty())),
            current_cost_model: revisions.cost_model,
            current_dynamic_obstacles: revisions.dynamic_obstacles,
            now_tick: 0,
            changed_blocks: blocks,
            changed_chunks: chunks,
        }
    }

    struct TestValidator {
        scope: WorldScope,
        revisions: RouteRevisions,
        cost: Cost,
        validation_tick: u64,
        reject: Option<(BlockPos, BlockPos)>,
        calls: Vec<(BlockPos, BlockPos, MoveKind)>,
    }

    impl PathEdgeValidator for TestValidator {
        fn scope(&self) -> &WorldScope {
            &self.scope
        }

        fn revisions(&self) -> RouteRevisions {
            self.revisions
        }

        fn validation_tick(&self) -> u64 {
            self.validation_tick
        }

        fn validate_edge(
            &mut self,
            from: BlockPos,
            to: BlockPos,
            reached_by: MoveKind,
        ) -> Result<Cost, EdgeValidationError> {
            self.calls.push((from, to, reached_by));
            if self.reject == Some((from, to)) {
                Err(EdgeValidationError::Unsafe)
            } else {
                Ok(self.cost)
            }
        }
    }

    fn validator(key: &WorldScope, revisions: RouteRevisions, cost: Cost) -> TestValidator {
        TestValidator {
            scope: key.clone(),
            revisions,
            cost,
            validation_tick: 0,
            reject: None,
            calls: Vec::new(),
        }
    }

    struct EmptyWorld;

    impl WorldView for EmptyWorld {
        fn block(&self, _pos: BlockPos) -> BlockKind {
            BlockKind::Air
        }
    }

    struct OneWideCorridor;

    impl WorldView for OneWideCorridor {
        fn block(&self, pos: BlockPos) -> BlockKind {
            if pos.z == 0 && (0..=2).contains(&pos.x) {
                if pos.y == 63 {
                    BlockKind::Solid
                } else if (64..=67).contains(&pos.y) {
                    BlockKind::Air
                } else {
                    BlockKind::Solid
                }
            } else {
                BlockKind::Solid
            }
        }
    }

    #[test]
    fn scope_is_explicit_and_bounded() {
        let key = scope("example.org:25565", "minecraft:overworld");
        assert_eq!(key.world(), "example.org:25565");
        assert_eq!(key.dimension(), "minecraft:overworld");
        assert!(matches!(
            WorldScope::try_new("", "overworld"),
            Err(WorldScopeError::Empty("world"))
        ));
        assert!(matches!(
            WorldScope::try_new("world", "bad\nkey"),
            Err(WorldScopeError::ContainsControl("dimension"))
        ));
        assert!(matches!(
            WorldScope::try_new("x".repeat(MAX_SCOPE_COMPONENT_BYTES + 1), "overworld"),
            Err(WorldScopeError::TooLong { .. })
        ));
    }

    #[test]
    fn overlay_expires_entries_and_never_exceeds_capacity() {
        let mut overlay = DynamicObstacleOverlay::try_new(scope("a", "overworld"), 2).unwrap();
        assert_eq!(overlay.revision(), 0);
        assert_eq!(
            overlay.insert(
                pos(1, 64, 0),
                OWNER,
                ObstacleKind::Hard,
                ObstaclePriority::NORMAL,
                10,
                5
            ),
            OverlayInsert::Inserted { evicted: None }
        );
        assert_eq!(overlay.revision(), 1);
        assert_eq!(
            overlay.insert(
                pos(2, 64, 0),
                OWNER,
                ObstacleKind::Soft { penalty: 50 },
                ObstaclePriority::NORMAL,
                10,
                5
            ),
            OverlayInsert::Inserted { evicted: None }
        );
        assert_eq!(overlay.advance_to(14), Ok(0));
        assert_eq!(overlay.active_len(), 2);
        assert_eq!(overlay.advance_to(15), Ok(2));
        assert_eq!(overlay.active_len(), 0);
        assert_eq!(overlay.retained_len(), 0);
        assert_eq!(overlay.revision(), 3);
    }

    #[test]
    fn capacity_preserves_priority_and_evicts_oldest_peer() {
        let mut overlay = DynamicObstacleOverlay::try_new(scope("a", "overworld"), 2).unwrap();
        let critical = pos(1, 64, 0);
        let old_normal = pos(2, 64, 0);
        let new_normal = pos(3, 64, 0);
        overlay.insert(
            critical,
            OWNER,
            ObstacleKind::Hard,
            ObstaclePriority::CRITICAL,
            0,
            100,
        );
        overlay.insert(
            old_normal,
            OWNER,
            ObstacleKind::Hard,
            ObstaclePriority::NORMAL,
            0,
            100,
        );
        assert_eq!(
            overlay.insert(
                pos(4, 64, 0),
                OWNER,
                ObstacleKind::Hard,
                ObstaclePriority::BACKGROUND,
                0,
                100
            ),
            OverlayInsert::RejectedCapacity
        );
        assert_eq!(
            overlay.insert(
                new_normal,
                OWNER,
                ObstacleKind::Hard,
                ObstaclePriority::NORMAL,
                0,
                100
            ),
            OverlayInsert::Inserted {
                evicted: Some(EvictedObstacle {
                    pos: old_normal,
                    owner: OWNER,
                })
            }
        );
        assert!(overlay.get(critical).is_some());
        assert!(overlay.get(old_normal).is_none());
        assert!(overlay.get(new_normal).is_some());
        assert_eq!(overlay.retained_len(), 2);
    }

    #[test]
    fn same_position_requires_equal_or_higher_priority() {
        let target = pos(1, 64, 0);
        let mut overlay = DynamicObstacleOverlay::try_new(scope("a", "overworld"), 1).unwrap();
        overlay.insert(
            target,
            OWNER,
            ObstacleKind::Hard,
            ObstaclePriority::CRITICAL,
            0,
            10,
        );
        assert_eq!(
            overlay.insert(
                target,
                OWNER,
                ObstacleKind::Soft { penalty: 10 },
                ObstaclePriority::NORMAL,
                1,
                10
            ),
            OverlayInsert::RejectedLowerPriority
        );
        assert_eq!(overlay.get(target).unwrap().kind(), ObstacleKind::Hard);
        assert_eq!(
            overlay.insert(
                target,
                OWNER,
                ObstacleKind::Soft { penalty: 20 },
                ObstaclePriority::CRITICAL,
                1,
                10
            ),
            OverlayInsert::RejectedHardDowngrade
        );
        assert_eq!(
            overlay.insert(
                target,
                ObstacleOwner(11),
                ObstacleKind::Hard,
                ObstaclePriority::CRITICAL,
                1,
                10
            ),
            OverlayInsert::RejectedOwnedByPeer
        );
    }

    #[test]
    fn invalid_ttls_and_zero_soft_penalties_are_rejected() {
        let mut overlay = DynamicObstacleOverlay::try_new(scope("a", "overworld"), 1).unwrap();
        assert_eq!(
            overlay.insert(
                pos(0, 0, 0),
                OWNER,
                ObstacleKind::Hard,
                ObstaclePriority::NORMAL,
                0,
                0
            ),
            OverlayInsert::RejectedInvalidTtl
        );
        assert_eq!(
            overlay.insert(
                pos(0, 0, 0),
                OWNER,
                ObstacleKind::Soft { penalty: 0 },
                ObstaclePriority::NORMAL,
                0,
                1
            ),
            OverlayInsert::RejectedZeroPenalty
        );
        assert_eq!(
            overlay.insert(
                pos(0, 0, 0),
                OWNER,
                ObstacleKind::Hard,
                ObstaclePriority::NORMAL,
                u64::MAX,
                1
            ),
            OverlayInsert::RejectedInvalidTtl
        );
    }

    #[test]
    fn apply_clones_context_and_combines_soft_costs_without_mutation() {
        let key = scope("a", "overworld");
        let soft = pos(1, 64, 0);
        let hard = pos(2, 64, 0);
        let mut overlay = DynamicObstacleOverlay::try_new(key.clone(), 2).unwrap();
        overlay.insert(
            soft,
            OWNER,
            ObstacleKind::Soft { penalty: 30 },
            ObstaclePriority::NORMAL,
            0,
            10,
        );
        overlay.insert(
            hard,
            OWNER,
            ObstacleKind::Hard,
            ObstaclePriority::NORMAL,
            0,
            10,
        );
        let mut original_avoid = HashMap::new();
        original_avoid.insert(soft, 7);
        let input = MoveContext {
            avoid: Arc::new(original_avoid),
            ..MoveContext::default()
        };
        let applied = overlay.apply(&key, 1, None, &EmptyWorld, &input).unwrap();
        assert_eq!(input.avoid.get(&soft), Some(&7));
        assert_eq!(input.avoid.get(&hard), None);
        assert_eq!(applied.context().avoid.get(&soft), Some(&37));
        assert_eq!(applied.context().avoid.get(&hard), Some(&Cost::MAX));
        assert_eq!(applied.block(hard), BlockKind::Fence);
        assert_eq!(applied.block(soft), BlockKind::Air);
        assert!(applied.is_hard(hard));
        assert!(!applied.standable(hard));
        assert!(!applied.standable(offset(hard, 0, 1, 0)));
        assert_eq!(applied.overlay_epoch().revision, overlay.revision());
    }

    #[test]
    fn hard_obstacle_cannot_be_used_as_a_floor_or_crossed_by_builtin_moves() {
        let key = scope("a", "overworld");
        let occupied = pos(1, 64, 0);
        let mut overlay = DynamicObstacleOverlay::try_new(key.clone(), 1).unwrap();
        overlay.insert(
            occupied,
            OWNER,
            ObstacleKind::Hard,
            ObstaclePriority::NORMAL,
            0,
            20,
        );
        let input = MoveContext {
            goal_tolerance: 0,
            ..MoveContext::default()
        };
        let applied = overlay
            .apply(&key, 1, None, &OneWideCorridor, &input)
            .unwrap();
        assert!(!applied.standable(occupied));
        assert!(!applied.standable(offset(occupied, 0, 1, 0)));
        assert!(
            crate::local::astar::find_path(
                &applied,
                pos(0, 64, 0),
                pos(2, 64, 0),
                &crate::local::moves::default_moves(),
                applied.context(),
            )
            .is_err()
        );
    }

    #[test]
    fn apply_rejects_cross_dimension_data() {
        let key = scope("a", "overworld");
        let mut overlay = DynamicObstacleOverlay::try_new(key, 1).unwrap();
        let error = overlay
            .apply(
                &scope("a", "the_nether"),
                0,
                None,
                &EmptyWorld,
                &MoveContext::default(),
            )
            .err()
            .expect("scope mismatch");
        assert!(matches!(
            error,
            OverlayApplyError::Scope(OverlayScopeMismatch { actual, .. })
                if actual.dimension() == "the_nether"
        ));
    }

    #[test]
    fn overlay_epoch_expires_and_advances_observable_revision() {
        let key = scope("a", "overworld");
        let target = pos(0, 64, 0);
        let mut overlay = DynamicObstacleOverlay::try_new(key.clone(), 1).unwrap();
        overlay.insert(
            target,
            OWNER,
            ObstacleKind::Hard,
            ObstaclePriority::NORMAL,
            10,
            5,
        );
        let before = overlay
            .apply(&key, 14, None, &EmptyWorld, &MoveContext::default())
            .unwrap();
        assert!(before.is_hard(target));
        assert_eq!(before.overlay_epoch().valid_until_tick, Some(15));
        let before_revision = before.overlay_epoch().revision;

        let after = overlay
            .apply(&key, 15, None, &EmptyWorld, &MoveContext::default())
            .unwrap();
        assert!(!after.is_hard(target));
        assert!(after.overlay_epoch().revision > before_revision);
        assert_eq!(overlay.retained_len(), 0);
    }

    #[test]
    fn owners_self_exclude_and_deadlocks_degrade_deterministically() {
        let key = scope("a", "overworld");
        let target = pos(0, 64, 0);
        let policy = DeadlockPolicy {
            max_hard_ticks: 5,
            yielded_soft_penalty: 777,
        };
        let mut overlay =
            DynamicObstacleOverlay::try_new_with_policy(key.clone(), 1, policy).unwrap();
        overlay.insert(
            target,
            OWNER,
            ObstacleKind::Hard,
            ObstaclePriority::NORMAL,
            0,
            20,
        );

        let own = overlay
            .apply(&key, 1, Some(OWNER), &EmptyWorld, &MoveContext::default())
            .unwrap();
        assert!(!own.is_hard(target));
        assert_eq!(own.context().avoid.get(&target), None);

        // Lower owner ID has deterministic right of way and sees a soft toll.
        let precedence = overlay
            .apply(
                &key,
                1,
                Some(ObstacleOwner(5)),
                &EmptyWorld,
                &MoveContext::default(),
            )
            .unwrap();
        assert!(!precedence.is_hard(target));
        assert_eq!(precedence.context().avoid.get(&target), Some(&777));

        let yielding = overlay
            .apply(
                &key,
                1,
                Some(ObstacleOwner(20)),
                &EmptyWorld,
                &MoveContext::default(),
            )
            .unwrap();
        assert!(yielding.is_hard(target));

        let timed_out = overlay
            .apply(
                &key,
                5,
                Some(ObstacleOwner(20)),
                &EmptyWorld,
                &MoveContext::default(),
            )
            .unwrap();
        assert!(!timed_out.is_hard(target));
        assert_eq!(timed_out.context().avoid.get(&target), Some(&777));
        assert_eq!(
            overlay.insert(
                pos(1, 64, 0),
                OWNER,
                ObstacleKind::Hard,
                ObstaclePriority::NORMAL,
                4,
                10
            ),
            OverlayInsert::RejectedClockRegression
        );
    }

    #[test]
    fn footprints_cover_support_head_corners_arc_landing_climb_and_hazards() {
        let diagonal = transition_dependency_footprint(
            0,
            pos(0, 64, 0),
            pos(1, 64, 1),
            MoveKind::Walk,
            DependencyOptions {
                hazard_radius: 2,
                ..DependencyOptions::default()
            },
        );
        assert!(diagonal.contains(pos(0, 63, 0)), "start support");
        assert!(diagonal.contains(pos(1, 65, 1)), "landing head");
        assert!(diagonal.contains(pos(1, 64, 0)), "diagonal corner");
        assert!(diagonal.contains(pos(0, 64, 1)), "other diagonal corner");
        assert!(diagonal.contains(pos(3, 64, 1)), "hazard radius");
        assert!(diagonal.contains(pos(2, 63, 1)), "landing neighborhood");

        let jump = transition_dependency_footprint(
            0,
            pos(0, 64, 0),
            pos(1, 65, 0),
            MoveKind::Jump,
            DependencyOptions {
                hazard_radius: 0,
                ..DependencyOptions::default()
            },
        );
        assert!(jump.contains(pos(0, 67, 0)), "jump arc headroom");

        let climb = transition_dependency_footprint(
            1,
            pos(5, 64, 5),
            pos(5, 65, 5),
            MoveKind::Climb,
            DependencyOptions {
                hazard_radius: 0,
                ..DependencyOptions::default()
            },
        );
        assert!(climb.contains(pos(6, 65, 5)), "ladder attachment");

        let swim = transition_dependency_footprint(
            2,
            pos(8, 64, 8),
            pos(8, 64, 9),
            MoveKind::Swim,
            DependencyOptions {
                hazard_radius: 0,
                local_neighborhood_radius: 0,
                ..DependencyOptions::default()
            },
        );
        assert!(swim.contains(pos(9, 64, 8)), "water neighborhood");

        let descending_parkour = transition_dependency_footprint(
            3,
            pos(0, 70, 0),
            pos(3, 60, 0),
            MoveKind::Parkour {
                blocks: 3,
                rise: -10,
            },
            DependencyOptions {
                hazard_radius: 0,
                local_neighborhood_radius: 0,
                ..DependencyOptions::default()
            },
        );
        assert!(
            descending_parkour.contains(pos(-2, 69, 0)),
            "two-block run-up"
        );
        assert!(
            descending_parkour.contains(pos(1, 61, 0)),
            "full descending arc corridor"
        );

        let fall = transition_dependency_footprint(
            4,
            pos(0, 70, 0),
            pos(1, 60, 0),
            MoveKind::Fall,
            DependencyOptions {
                hazard_radius: 0,
                local_neighborhood_radius: 0,
                ..DependencyOptions::default()
            },
        );
        assert!(
            fall.contains(pos(1, 65, 0)),
            "fall must track the landing column"
        );
    }

    #[test]
    fn oversized_or_long_range_footprints_fail_safe_to_broad() {
        let bounded = transition_dependency_footprint(
            0,
            pos(0, 64, 0),
            pos(1, 64, 0),
            MoveKind::Walk,
            DependencyOptions {
                hazard_radius: MAX_LAVA_SCAN_RADIUS,
                max_blocks_per_edge: 8,
                ..DependencyOptions::default()
            },
        );
        assert!(bounded.invalidates_on_any_change());
        assert!(bounded.blocks().len() <= 8);

        let long = transition_dependency_footprint(
            0,
            pos(0, 64, 0),
            pos(1_000, 64, 0),
            MoveKind::Aotv,
            DependencyOptions::default(),
        );
        assert!(long.invalidates_on_any_change());

        let generated = DependencyFootprint::from_generator_blocks(
            0,
            pos(0, 64, 0),
            pos(1, 64, 0),
            [pos(0, 63, 0), pos(1, 63, 0)],
        );
        assert!(!generated.invalidates_on_any_change());
        assert!(generated.contains(pos(1, 63, 0)));
        let empty = DependencyFootprint::from_generator_blocks(0, pos(0, 64, 0), pos(1, 64, 0), []);
        assert!(empty.invalidates_on_any_change());
    }

    #[test]
    fn invalidation_returns_earliest_edge_and_ignores_unrelated_changes() {
        let key = scope("a", "overworld");
        let revisions = revisions(4, 7, 9, 2);
        let route = path(&[0, 1, 2], 20);
        let first = DependencyFootprint {
            edge_index: 0,
            from: pos(0, 64, 0),
            to: pos(1, 64, 0),
            blocks: vec![pos(10, 64, 0)],
            invalidate_on_any_change: false,
        };
        let second = DependencyFootprint {
            edge_index: 1,
            from: pos(1, 64, 0),
            to: pos(2, 64, 0),
            blocks: vec![pos(10, 64, 0), pos(20, 64, 0)],
            invalidate_on_any_change: false,
        };
        let mut index =
            PathDependencyIndex::from_footprints(&route, key.clone(), revisions, &[first, second])
                .unwrap();
        let second_only = [pos(20, 64, 0)];
        assert_eq!(
            index
                .earliest_affected(&batch(&key, revisions, &second_only, &[]))
                .unwrap(),
            Some(1)
        );
        let shared = [pos(10, 64, 0)];
        assert_eq!(
            index
                .earliest_affected(&batch(&key, revisions, &shared, &[]))
                .unwrap(),
            Some(0)
        );
        let unrelated = [pos(999, 64, 0)];
        assert_eq!(
            index
                .earliest_affected(&batch(&key, revisions, &unrelated, &[]))
                .unwrap(),
            None
        );
        let advanced = index.revisions();
        assert_eq!(
            index
                .earliest_affected(&batch(&key, advanced, &[], &[]))
                .unwrap(),
            None
        );
        let next_basis = index.revisions();
        let second_unrelated = [pos(-999, 70, -999)];
        assert_eq!(
            index
                .earliest_affected(&batch(&key, next_basis, &second_unrelated, &[]))
                .unwrap(),
            None
        );
        assert_eq!(index.revisions().world_basis, next_basis.world_basis + 1);
    }

    #[test]
    fn invalidation_rejects_scope_and_revision_mismatches() {
        let key = scope("a", "overworld");
        let revisions = revisions(1, 2, 3, 4);
        let route = path(&[0, 1], 10);
        let mut index = PathDependencyIndex::build_unknown(&route, key.clone(), revisions);
        let other_scope = scope("a", "nether");
        let other_batch = batch(&other_scope, revisions, &[], &[]);
        assert!(matches!(
            index.earliest_affected(&other_batch),
            Err(InvalidationError::ScopeMismatch { .. })
        ));
        let mut wrong_path = batch(&key, revisions, &[], &[]);
        wrong_path.path_revision = 9;
        assert!(matches!(
            index.earliest_affected(&wrong_path),
            Err(InvalidationError::PathRevisionMismatch { .. })
        ));
        let mut wrong_basis = batch(&key, revisions, &[], &[]);
        wrong_basis.from_world_revision = 1;
        assert!(matches!(
            index.earliest_affected(&wrong_basis),
            Err(InvalidationError::WorldContinuityMismatch { .. })
        ));

        let mut changed_cost = batch(&key, revisions, &[], &[]);
        changed_cost.current_cost_model += 1;
        assert_eq!(index.earliest_affected(&changed_cost).unwrap(), Some(0));

        let changed = [pos(0, 64, 0)];
        let mut missing_revision = batch(&key, revisions, &changed, &[]);
        missing_revision.to_world_revision = missing_revision.from_world_revision;
        assert!(matches!(
            index.earliest_affected(&missing_revision),
            Err(InvalidationError::WorldChangeWithoutRevision { .. })
        ));
    }

    #[test]
    fn broad_dependency_invalidates_on_any_nonempty_change() {
        let key = scope("a", "overworld");
        let revisions = revisions(1, 1, 1, 1);
        let route = path(&[0, 1, 2, 3], 30);
        let first = DependencyFootprint {
            edge_index: 0,
            from: route.nodes[0].pos,
            to: route.nodes[1].pos,
            blocks: vec![pos(10, 0, 0)],
            invalidate_on_any_change: false,
        };
        let second = DependencyFootprint {
            edge_index: 1,
            from: route.nodes[1].pos,
            to: route.nodes[2].pos,
            blocks: vec![pos(20, 0, 0)],
            invalidate_on_any_change: false,
        };
        let broad = DependencyFootprint {
            edge_index: 2,
            from: route.nodes[2].pos,
            to: route.nodes[3].pos,
            blocks: vec![],
            invalidate_on_any_change: true,
        };
        let mut index = PathDependencyIndex::from_footprints(
            &route,
            key.clone(),
            revisions,
            &[first, second, broad],
        )
        .unwrap();
        let unrelated = [pos(-999, 5, 42)];
        assert_eq!(
            index
                .earliest_affected(&batch(&key, revisions, &unrelated, &[]))
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            index
                .earliest_affected(&batch(&key, revisions, &[], &[]))
                .unwrap(),
            None
        );
    }

    #[test]
    fn dependency_index_built_from_path_catches_support_and_ignores_far_block() {
        let route = path(&[0, 1, 2], 20);
        let key = scope("a", "overworld");
        let revisions = revisions(1, 1, 1, 1);
        let ctx = MoveContext {
            lava_policy: LavaPolicy::Penalized,
            lava_proximity_radius: 0,
            ..MoveContext::default()
        };
        let mut index = PathDependencyIndex::build_builtin(&route, key.clone(), revisions, &ctx);
        let support = [pos(1, 63, 0)];
        assert_eq!(
            index
                .earliest_affected(&batch(&key, revisions, &support, &[]))
                .unwrap(),
            Some(0)
        );
        let far = [pos(100, 64, 100)];
        assert_eq!(
            index
                .earliest_affected(&batch(&key, revisions, &far, &[]))
                .unwrap(),
            None
        );

        let loaded = [ChunkChange {
            column: ChunkColumn::containing(pos(1, 64, 0)),
            kind: ChunkChangeKind::Loaded,
        }];
        let advanced = index.revisions();
        assert_eq!(
            index
                .earliest_affected(&batch(&key, advanced, &[], &loaded))
                .unwrap(),
            Some(0)
        );
    }

    #[test]
    fn missing_or_endpoint_mismatched_footprints_fail_broad() {
        let route = path(&[0, 1, 2], 20);
        let key = scope("a", "overworld");
        let revisions = revisions(1, 1, 1, 1);
        let edge_zero = DependencyFootprint {
            edge_index: 0,
            from: route.nodes[0].pos,
            to: route.nodes[1].pos,
            blocks: vec![pos(10, 64, 0)],
            invalidate_on_any_change: false,
        };
        let mut incomplete = PathDependencyIndex::from_footprints(
            &route,
            key.clone(),
            revisions,
            std::slice::from_ref(&edge_zero),
        )
        .unwrap();
        let unrelated = [pos(999, 64, 999)];
        assert_eq!(
            incomplete
                .earliest_affected(&batch(&key, revisions, &unrelated, &[]))
                .unwrap(),
            Some(1)
        );

        let wrong_endpoint = DependencyFootprint {
            to: pos(99, 64, 0),
            ..edge_zero
        };
        let mut mismatched =
            PathDependencyIndex::from_footprints(&route, key.clone(), revisions, &[wrong_endpoint])
                .unwrap();
        assert_eq!(
            mismatched
                .earliest_affected(&batch(&key, revisions, &unrelated, &[]))
                .unwrap(),
            Some(0)
        );
    }

    #[test]
    fn unknown_and_teleport_paths_are_broad_and_short_circuit_work() {
        let key = scope("a", "overworld");
        let revisions = revisions(1, 1, 1, 1);
        let custom = path(&[0, 1, 2, 3], 30);
        let unknown = PathDependencyIndex::build_unknown(&custom, key.clone(), revisions);
        assert_eq!(unknown.indexed_blocks(), 0);

        let mut teleport_nodes = Vec::with_capacity(10_001);
        teleport_nodes.push(node(0, MoveKind::Start));
        teleport_nodes.push(node(50, MoveKind::Aotv));
        for x in 51..=10_049 {
            teleport_nodes.push(node(x, MoveKind::Walk));
        }
        let teleport = Path {
            nodes: teleport_nodes,
            total_cost: 10,
        };
        let mut inferred = PathDependencyIndex::build_builtin(
            &teleport,
            key.clone(),
            revisions,
            &MoveContext::default(),
        );
        assert_eq!(inferred.edge_count(), 10_000);
        assert_eq!(inferred.indexed_blocks(), 0);
        let far = [pos(-10_000, 5, 10_000)];
        assert_eq!(
            inferred
                .earliest_affected(&batch(&key, revisions, &far, &[]))
                .unwrap(),
            Some(0)
        );
    }

    #[test]
    fn overlay_expiry_and_unexplained_world_revision_invalidate_from_start() {
        let key = scope("a", "overworld");
        let mut revisions = revisions(1, 7, 1, 1);
        revisions.dynamic_obstacles.valid_until_tick = Some(20);
        let route = path(&[0, 1], 10);
        let mut index = PathDependencyIndex::build_unknown(&route, key.clone(), revisions);

        let mut expired = batch(&key, revisions, &[], &[]);
        expired.now_tick = 20;
        assert_eq!(index.earliest_affected(&expired).unwrap(), Some(0));

        let mut unspecified = batch(&key, revisions, &[], &[]);
        unspecified.to_world_revision += 1;
        assert_eq!(index.earliest_affected(&unspecified).unwrap(), Some(0));
    }

    #[test]
    fn complete_paths_join_only_at_matching_endpoints() {
        let first = path(&[0, 1, 2], 20);
        let second = path(&[2, 3, 4], 30);
        let joined = structurally_join_paths(&[&first, &second]).unwrap();
        assert_eq!(
            joined
                .nodes()
                .iter()
                .map(|node| node.pos.x)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );

        let mismatch = path(&[9, 10], 10);
        assert!(matches!(
            structurally_join_paths(&[&first, &mismatch]),
            Err(PathSpliceError::EndpointMismatch { .. })
        ));
    }

    #[test]
    fn splice_revalidates_every_edge_and_regenerates_exact_cost() {
        let original = path(&[0, 1, 2, 3, 4], 40);
        let patch = path(&[1, 8, 9, 3], 25);
        let key = scope("a", "overworld");
        let revisions = revisions(1, 1, 1, 1);
        let mut edge_validator = validator(&key, revisions, 7);
        let spliced = splice_path_validated(
            &original,
            1,
            3,
            &patch,
            &key,
            revisions,
            &mut edge_validator,
        )
        .unwrap();
        assert_eq!(
            spliced
                .path()
                .nodes
                .iter()
                .map(|node| node.pos.x)
                .collect::<Vec<_>>(),
            vec![0, 1, 8, 9, 3, 4]
        );
        assert_eq!(spliced.path().total_cost, 35);
        assert_eq!(edge_validator.calls.len(), 5);
        assert_eq!(spliced.path().nodes[0].reached_by, MoveKind::Start);

        assert!(matches!(
            structurally_splice_path(&original, 1, 3, &path(&[2, 8, 3], 10)),
            Err(PathSpliceError::EndpointMismatch { .. })
        ));
        assert!(matches!(
            structurally_splice_path(&original, 3, 1, &patch),
            Err(PathSpliceError::InvalidRange { .. })
        ));

        let mut rejecting = validator(&key, revisions, 1);
        rejecting.reject = Some((pos(8, 64, 0), pos(9, 64, 0)));
        assert!(matches!(
            splice_path_validated(&original, 1, 3, &patch, &key, revisions, &mut rejecting),
            Err(PathSpliceError::EdgeRejected {
                reason: EdgeValidationError::Unsafe,
                ..
            })
        ));
    }

    #[test]
    fn malformed_paths_scope_revisions_and_cost_overflow_are_rejected() {
        let malformed = Path {
            nodes: vec![node(0, MoveKind::Walk), node(1, MoveKind::Walk)],
            total_cost: 10,
        };
        assert!(matches!(
            structurally_join_paths(&[&malformed]),
            Err(PathSpliceError::InvalidStartMarker { .. })
        ));
        let embedded = Path {
            nodes: vec![node(0, MoveKind::Start), node(1, MoveKind::Start)],
            total_cost: 10,
        };
        assert!(matches!(
            structurally_join_paths(&[&embedded]),
            Err(PathSpliceError::EmbeddedStart { .. })
        ));

        let key = scope("a", "overworld");
        let revisions = revisions(1, 1, 1, 1);
        let structural = structurally_join_paths(&[&path(&[0, 1], 0), &path(&[1, 2], 0)]).unwrap();
        let mut expensive = validator(&key, revisions, Cost::MAX);
        assert!(matches!(
            validate_structural_path(structural, &key, revisions, &mut expensive),
            Err(PathSpliceError::CostOverflow)
        ));

        let structural = structurally_join_paths(&[&path(&[0, 1], 0)]).unwrap();
        let mut wrong_scope = validator(&scope("b", "overworld"), revisions, 1);
        assert!(matches!(
            validate_structural_path(structural, &key, revisions, &mut wrong_scope),
            Err(PathSpliceError::ValidatorScopeMismatch { .. })
        ));

        let structural = structurally_join_paths(&[&path(&[0, 1], 0)]).unwrap();
        let mut wrong_revision = validator(
            &key,
            RouteRevisions {
                cost_model: 2,
                ..revisions
            },
            1,
        );
        assert!(matches!(
            validate_structural_path(structural, &key, revisions, &mut wrong_revision),
            Err(PathSpliceError::ValidatorRevisionMismatch { .. })
        ));

        let mut expiring_revisions = revisions;
        expiring_revisions.dynamic_obstacles.valid_until_tick = Some(5);
        let structural = structurally_join_paths(&[&path(&[0, 1], 0)]).unwrap();
        let mut expired_validator = validator(&key, expiring_revisions, 1);
        expired_validator.validation_tick = 5;
        assert!(matches!(
            validate_structural_path(structural, &key, expiring_revisions, &mut expired_validator),
            Err(PathSpliceError::ValidatorEpochExpired { .. })
        ));

        let single = path(&[0], 999);
        let structural = structurally_join_paths(&[&single]).unwrap();
        let mut no_edges = validator(&key, revisions, 7);
        let validated =
            validate_structural_path(structural, &key, revisions, &mut no_edges).unwrap();
        assert_eq!(validated.path().total_cost, 0);
    }
}
