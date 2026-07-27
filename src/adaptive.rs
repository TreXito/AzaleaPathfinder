//! Bounded telemetry and conservative online calibration.
//!
//! Learning is deliberately separated into three stages:
//!
//! 1. execution produces an exactly-once [`MoveObservation`];
//! 2. an [`AdaptiveProfile`] builds a shadow candidate from bounded,
//!    context-specific relative multipliers;
//! 3. an explicit non-regression report promotes that candidate to the
//!    known-good model used by future navigation legs.
//!
//! Feasibility and safety constraints are not represented in the learned
//! model, so calibration cannot make lava legal, increase a survivable fall,
//! or relax collision validation.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use atomic_write_file::AtomicWriteFile;
use azalea::BlockPos;
use serde::{Deserialize, Serialize};

use crate::local::follower::FollowerSettings;
use crate::{Cost, LavaPolicy, MoveContext, MovementCosts, WaterPolicy};

pub const ADAPTIVE_SCHEMA_VERSION: u16 = 4;
pub const FEATURE_SCHEMA_VERSION: u16 = 2;
pub const DEFAULT_MIN_SAMPLES: u32 = 32;
pub const MAX_PROFILE_BUCKETS: usize = 256;
pub const MAX_RECENT_COMPLETIONS: usize = 4_096;
pub const MAX_PROFILE_BYTES: u64 = 1_048_576;
pub const MAX_TELEMETRY_FILE_BYTES: u64 = 8 * 1_048_576;
const MAX_CONTEXT_TEXT_BYTES: usize = 128;
const MAX_OBSERVED_TICKS: u64 = 20_000;
const MAX_OBSERVATION_NONCE: u64 = u64::MAX - 2;
const MULTIPLIER_ONE: u32 = 1_000_000;
const MIN_SAMPLE_MULTIPLIER: u32 = 250_000;
const MAX_SAMPLE_MULTIPLIER: u32 = 4_000_000;

/// Stable persisted motion-primitive identifier.
///
/// Values are explicit and independent of Rust enum ordering or debug
/// formatting. Built-in generators author this ID; consumers must not infer it
/// from two path nodes because `Swim`, for example, covers both a vertical
/// stroke and a bank exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PrimitiveId(pub u16);

impl PrimitiveId {
    pub const WALK_CARDINAL: Self = Self(1);
    pub const WALK_DIAGONAL: Self = Self(2);
    pub const STEP: Self = Self(3);
    pub const JUMP: Self = Self(4);
    pub const FALL: Self = Self(5);
    pub const PARKOUR: Self = Self(6);
    pub const CLIMB: Self = Self(7);
    pub const SWIM: Self = Self(8);
    pub const SWIM_EXIT: Self = Self(9);
    pub const AOTV: Self = Self(10);
    pub const ETHERWARP: Self = Self(11);

    pub const fn is_builtin(self) -> bool {
        self.0 >= Self::WALK_CARDINAL.0 && self.0 <= Self::ETHERWARP.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TerrainClass {
    Unknown = 0,
    FullBlock = 1,
    PartialBlock = 2,
    Climbable = 3,
    Water = 4,
}

/// Versioned, bounded features authored with a planned edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveFeatures {
    pub schema: u16,
    /// Generator-defined travel distance. For parkour this is the
    /// `MoveKind::Parkour::blocks` value, not Chebyshev distance.
    pub horizontal_blocks: u8,
    pub vertical_blocks: i8,
    pub adjacent_walls: u8,
    pub terrain: TerrainClass,
    pub flags: u16,
}

impl MoveFeatures {
    pub const FLAG_LOW_HEADROOM: u16 = 1 << 1;
    pub const FLAG_FRACTIONAL_NEIGHBOR: u16 = 1 << 2;
    pub const FLAG_WATER_ENTRY: u16 = 1 << 3;
    pub const KNOWN_FLAGS: u16 =
        Self::FLAG_LOW_HEADROOM | Self::FLAG_FRACTIONAL_NEIGHBOR | Self::FLAG_WATER_ENTRY;

    pub const fn new(horizontal_blocks: u8, vertical_blocks: i8, terrain: TerrainClass) -> Self {
        Self {
            schema: FEATURE_SCHEMA_VERSION,
            horizontal_blocks,
            vertical_blocks,
            adjacent_walls: 0,
            terrain,
            flags: 0,
        }
    }

    pub const fn with_surroundings(mut self, adjacent_walls: u8, flags: u16) -> Self {
        self.adjacent_walls = adjacent_walls;
        self.flags = flags;
        self
    }

    fn validate(self) -> bool {
        self.schema == FEATURE_SCHEMA_VERSION
            && self.horizontal_blocks <= 16
            && self.adjacent_walls <= 8
            && self.flags & !Self::KNOWN_FLAGS == 0
    }
}

/// Context that prevents observations from unrelated physics regimes from
/// contaminating one another.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProfileKey {
    /// Explicit operator-provided server/profile key. It is never inferred.
    pub server: String,
    pub protocol: u32,
    pub dimension: String,
    pub capability_hash: u64,
    pub latency_bucket_ms: u16,
    /// Manual operator metadata such as an anticheat policy.
    pub environment: String,
}

impl ProfileKey {
    /// A safe test/offline key. Production callers should provide their actual
    /// server and dimension before enabling adaptation.
    pub fn local_default() -> Self {
        Self {
            server: "local-test-only".into(),
            protocol: 0,
            dimension: "unknown".into(),
            capability_hash: 0,
            latency_bucket_ms: 0,
            environment: "shadow".into(),
        }
    }

    pub fn validate(&self) -> bool {
        [
            self.server.as_str(),
            self.dimension.as_str(),
            self.environment.as_str(),
        ]
        .into_iter()
        .all(valid_context_text)
    }

    /// Compatibility-sized digest for logs and seeds. File names use the full
    /// 256-bit digest returned by the private canonical encoder.
    pub fn stable_hash(&self) -> u64 {
        let digest = self.digest();
        u64::from_le_bytes(
            digest.as_bytes()[..8]
                .try_into()
                .expect("eight digest bytes"),
        )
    }

    fn digest(&self) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"azalea-pathfinder/profile-key/v1");
        hash_text(&mut hasher, &self.server);
        hasher.update(&self.protocol.to_le_bytes());
        hash_text(&mut hasher, &self.dimension);
        hasher.update(&self.capability_hash.to_le_bytes());
        hasher.update(&self.latency_bucket_ms.to_le_bytes());
        hash_text(&mut hasher, &self.environment);
        hasher.finalize()
    }
}

fn hash_text(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u32).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn valid_context_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_CONTEXT_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterruptionReason {
    Pause,
    Cancelled,
    GoalChanged,
    WorldChanged,
    ExternalImpulse,
    ServerCorrection,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationOutcome {
    Success,
    Stalled,
    Unsafe,
    FellOff,
    TimedOut,
    Interrupted(InterruptionReason),
}

impl ObservationOutcome {
    fn updates_failure_model(self) -> bool {
        matches!(
            self,
            Self::Stalled | Self::Unsafe | Self::FellOff | Self::TimedOut
        )
    }
}

/// Evidence supplied by the executor that explains why an attempt ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionEvidence {
    ReachedPlannedNode,
    StallDetector,
    LiveValidationFailure,
    FellBelowPath,
    CumulativeDeadline,
    PauseOrCancellation,
    ExternalMovementEvent,
}

/// Immutable metadata frozen when one navigation leg is planned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationContext {
    pub profile: ProfileKey,
    pub journey_id: u64,
    pub generation: u64,
    pub leg: u32,
    pub plan_revision: u64,
    pub world_revision: u64,
    pub model_revision: u64,
    pub planner_settings_hash: u64,
    pub control_settings_hash: u64,
    pub baseline_costs_hash: u64,
    pub actor_capability_hash: u64,
    pub build_id: String,
    pub created_unix_ms: u64,
}

impl ObservationContext {
    pub fn validate(&self) -> bool {
        self.profile.validate()
            && self.journey_id != 0
            && self.planner_settings_hash != 0
            && self.control_settings_hash != 0
            && self.baseline_costs_hash != 0
            && self.actor_capability_hash == self.profile.capability_hash
            && valid_context_text(&self.build_id)
    }

    fn regime(&self) -> AdaptiveRegime {
        AdaptiveRegime {
            build_id: self.build_id.clone(),
            planner_settings_hash: self.planner_settings_hash,
            control_settings_hash: self.control_settings_hash,
            baseline_costs_hash: self.baseline_costs_hash,
            actor_capability_hash: self.actor_capability_hash,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObservationId {
    /// Process-unique, time-seeded monotonic anti-replay nonce.
    pub nonce: u64,
    pub journey_id: u64,
    pub generation: u64,
    pub leg: u32,
    pub edge_index: u32,
    pub attempt: u16,
}

fn observation_nonce_counter() -> &'static AtomicU64 {
    static NONCE: OnceLock<AtomicU64> = OnceLock::new();
    NONCE.get_or_init(|| {
        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |duration| {
                duration.as_millis().min(u128::from(u64::MAX >> 16)) as u64
            });
        let seed = (millis << 16) | u64::from(std::process::id() & 0xffff);
        AtomicU64::new(seed.max(1))
    })
}

pub fn next_observation_nonce() -> u64 {
    observation_nonce_counter()
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, nonce_successor)
        // Zero is invalid and therefore fails observation creation closed if
        // the process ever exhausts the 64-bit nonce space.
        .unwrap_or(0)
}

fn nonce_successor(current: u64) -> Option<u64> {
    (current < MAX_OBSERVATION_NONCE).then(|| current + 1)
}

fn ensure_next_observation_nonce_after(high_water: u64) -> Result<(), ProfileError> {
    if high_water >= MAX_OBSERVATION_NONCE {
        return Err(ProfileError::NonceExhausted);
    }
    let next = high_water
        .checked_add(1)
        .ok_or(ProfileError::NonceExhausted)?;
    observation_nonce_counter()
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            Some(current.max(next))
        })
        .expect("nonce update closure always succeeds");
    Ok(())
}

/// A begun primitive. This type is intentionally not `Clone`; finishing it
/// consumes it, making duplicate completion impossible in the owning executor.
#[derive(Debug)]
pub struct MoveAttempt {
    context: ObservationContext,
    id: ObservationId,
    primitive: PrimitiveId,
    features: MoveFeatures,
    from: BlockPos,
    to: BlockPos,
    predicted: CostComponents,
    predicted_ticks: u32,
    began_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationError {
    InvalidContext,
    InvalidPrimitive,
    InvalidFeatures,
    InvalidIdentity,
    InvalidPrediction,
    TickOrder,
    InvalidAttribution,
}

impl std::fmt::Display for ObservationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid movement observation: {self:?}")
    }
}

impl std::error::Error for ObservationError {}

impl MoveAttempt {
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        context: ObservationContext,
        id: ObservationId,
        primitive: PrimitiveId,
        features: MoveFeatures,
        from: BlockPos,
        to: BlockPos,
        predicted: CostComponents,
        predicted_ticks: u32,
        began_tick: u64,
    ) -> Result<Self, ObservationError> {
        if !context.validate() {
            return Err(ObservationError::InvalidContext);
        }
        if !primitive.is_builtin() {
            return Err(ObservationError::InvalidPrimitive);
        }
        if !features.validate() {
            return Err(ObservationError::InvalidFeatures);
        }
        if id.journey_id != context.journey_id
            || id.generation != context.generation
            || id.leg != context.leg
            || id.nonce == 0
            || id.nonce >= MAX_OBSERVATION_NONCE
        {
            return Err(ObservationError::InvalidIdentity);
        }
        if predicted.total() == 0 || predicted_ticks == 0 {
            return Err(ObservationError::InvalidPrediction);
        }
        Ok(Self {
            context,
            id,
            primitive,
            features,
            from,
            to,
            predicted,
            predicted_ticks,
            began_tick,
        })
    }

    pub fn finish(
        self,
        ended_tick: u64,
        completion_sequence: u64,
        outcome: ObservationOutcome,
        evidence: AttributionEvidence,
    ) -> Result<MoveObservation, ObservationError> {
        let actual_ticks = ended_tick
            .checked_sub(self.began_tick)
            .ok_or(ObservationError::TickOrder)?;
        if actual_ticks > MAX_OBSERVED_TICKS
            || (matches!(outcome, ObservationOutcome::Success) && actual_ticks == 0)
        {
            return Err(ObservationError::TickOrder);
        }
        if !attribution_matches(outcome, evidence) {
            return Err(ObservationError::InvalidAttribution);
        }
        let observation = MoveObservation {
            schema: ADAPTIVE_SCHEMA_VERSION,
            context: self.context,
            id: self.id,
            primitive: self.primitive,
            features: self.features,
            from: [self.from.x, self.from.y, self.from.z],
            to: [self.to.x, self.to.y, self.to.z],
            predicted: self.predicted,
            predicted_ticks: self.predicted_ticks,
            began_tick: self.began_tick,
            ended_tick,
            completion_sequence,
            outcome,
            evidence,
        };
        debug_assert!(observation.validate());
        Ok(observation)
    }
}

fn attribution_matches(outcome: ObservationOutcome, evidence: AttributionEvidence) -> bool {
    matches!(
        (outcome, evidence),
        (
            ObservationOutcome::Success,
            AttributionEvidence::ReachedPlannedNode
        ) | (
            ObservationOutcome::Stalled,
            AttributionEvidence::StallDetector
        ) | (
            ObservationOutcome::Unsafe,
            AttributionEvidence::LiveValidationFailure
        ) | (
            ObservationOutcome::FellOff,
            AttributionEvidence::FellBelowPath
        ) | (
            ObservationOutcome::TimedOut,
            AttributionEvidence::CumulativeDeadline
        ) | (
            ObservationOutcome::Interrupted(
                InterruptionReason::Pause
                    | InterruptionReason::Cancelled
                    | InterruptionReason::GoalChanged
            ),
            AttributionEvidence::PauseOrCancellation
        ) | (
            ObservationOutcome::Interrupted(
                InterruptionReason::WorldChanged
                    | InterruptionReason::ExternalImpulse
                    | InterruptionReason::ServerCorrection
                    | InterruptionReason::Unknown
            ),
            AttributionEvidence::ExternalMovementEvent
        )
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveObservation {
    pub schema: u16,
    pub context: ObservationContext,
    pub id: ObservationId,
    pub primitive: PrimitiveId,
    pub features: MoveFeatures,
    pub from: [i32; 3],
    pub to: [i32; 3],
    pub predicted: CostComponents,
    pub predicted_ticks: u32,
    pub began_tick: u64,
    pub ended_tick: u64,
    pub completion_sequence: u64,
    pub outcome: ObservationOutcome,
    pub evidence: AttributionEvidence,
}

impl MoveObservation {
    pub fn validate(&self) -> bool {
        self.schema == ADAPTIVE_SCHEMA_VERSION
            && self.context.validate()
            && self.id.journey_id == self.context.journey_id
            && self.id.generation == self.context.generation
            && self.id.leg == self.context.leg
            && self.id.nonce != 0
            && self.id.nonce < MAX_OBSERVATION_NONCE
            && self.primitive.is_builtin()
            && self.features.validate()
            && self.predicted.total() > 0
            && self.predicted_ticks > 0
            && self.ended_tick >= self.began_tick
            && self.ended_tick - self.began_tick <= MAX_OBSERVED_TICKS
            && (!matches!(self.outcome, ObservationOutcome::Success)
                || self.ended_tick > self.began_tick)
            && attribution_matches(self.outcome, self.evidence)
    }

    pub fn actual_ticks(&self) -> u64 {
        self.ended_tick.saturating_sub(self.began_tick)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostComponents {
    pub time: Cost,
    pub reliability: Cost,
    /// Immutable feasibility/policy toll. The learner never scales this.
    pub safety: Cost,
    pub damage: Cost,
    pub desync: Cost,
    pub resource: Cost,
}

impl CostComponents {
    pub fn total(self) -> Cost {
        self.time
            .saturating_add(self.reliability)
            .saturating_add(self.safety)
            .saturating_add(self.damage)
            .saturating_add(self.desync)
            .saturating_add(self.resource)
    }
}

/// Packed, version-stable context key. Numeric keys remain valid JSON object
/// keys and avoid serialization based on enum names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
struct FeatureKey(u64);

impl FeatureKey {
    fn fallback(primitive: PrimitiveId) -> Self {
        Self(u64::from(primitive.0))
    }

    fn exact(primitive: PrimitiveId, features: MoveFeatures) -> Self {
        let vertical = features.vertical_blocks as u8;
        let walls = features.adjacent_walls.min(4);
        let mut value = u64::from(primitive.0);
        value |= u64::from(features.horizontal_blocks) << 16;
        value |= u64::from(vertical) << 21;
        value |= u64::from(walls) << 29;
        value |= (features.terrain as u64) << 32;
        value |= u64::from(features.flags) << 36;
        value |= 1_u64 << 63;
        Self(value)
    }

    fn is_exact(self) -> bool {
        self.0 & (1_u64 << 63) != 0
    }

    fn validate(self) -> bool {
        let primitive = PrimitiveId((self.0 & 0xffff) as u16);
        if !primitive.is_builtin() {
            return false;
        }
        if !self.is_exact() {
            return self == Self::fallback(primitive);
        }
        let horizontal = (self.0 >> 16) & 0x1f;
        let walls = (self.0 >> 29) & 0x7;
        let terrain = (self.0 >> 32) & 0xf;
        let flags = ((self.0 >> 36) & 0xffff) as u16;
        let reserved = (self.0 >> 52) & 0x7ff;
        horizontal <= 16
            && walls <= 4
            && terrain <= TerrainClass::Water as u64
            && flags & !MoveFeatures::KNOWN_FLAGS == 0
            && reserved == 0
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PrimitiveEstimate {
    observations: u32,
    successes: u32,
    failures: u32,
    interruptions: u32,
    /// EWMA of `actual_ticks / predicted_ticks`, in millionths.
    ratio_ppm: u32,
    deviation_ppm: u32,
    last_sequence: u64,
}

impl PrimitiveEstimate {
    fn trials(&self) -> u32 {
        self.successes.saturating_add(self.failures)
    }

    fn observe(&mut self, observation: &MoveObservation, arrival_sequence: u64) {
        self.observations = self.observations.saturating_add(1);
        self.last_sequence = arrival_sequence;
        match observation.outcome {
            ObservationOutcome::Success => {
                self.successes = self.successes.saturating_add(1);
                let raw = observation
                    .actual_ticks()
                    .saturating_mul(u64::from(MULTIPLIER_ONE))
                    .checked_div(u64::from(observation.predicted_ticks))
                    .unwrap_or(u64::MAX);
                let sample = u32::try_from(raw)
                    .unwrap_or(u32::MAX)
                    .clamp(MIN_SAMPLE_MULTIPLIER, MAX_SAMPLE_MULTIPLIER);
                if self.ratio_ppm == 0 {
                    self.ratio_ppm = sample;
                } else {
                    let old = self.ratio_ppm;
                    self.ratio_ppm = ((u64::from(old) * 7 + u64::from(sample)) / 8) as u32;
                    let residual = old.abs_diff(sample);
                    self.deviation_ppm =
                        ((u64::from(self.deviation_ppm) * 7 + u64::from(residual)) / 8) as u32;
                }
            }
            outcome if outcome.updates_failure_model() => {
                self.failures = self.failures.saturating_add(1);
            }
            ObservationOutcome::Interrupted(_) => {
                self.interruptions = self.interruptions.saturating_add(1);
            }
            _ => {}
        }
    }

    fn proposed_multiplier(&self, settings: &AdaptiveSettings) -> Option<u32> {
        let trials = self.trials();
        if trials < settings.min_samples || self.successes < 3 || self.ratio_ppm == 0 {
            return None;
        }

        // One-sided Hoeffding bound with exp(-3) confidence, calculated in
        // fixed point. `sqrt(1.5/n) * 1e6`.
        let empirical_failure = u64::from(self.failures).saturating_mul(u64::from(MULTIPLIER_ONE))
            / u64::from(trials.max(1));
        let confidence = integer_sqrt(
            1_500_000_000_000_u64
                .checked_div(u64::from(trials.max(1)))
                .unwrap_or(u64::MAX),
        );
        let failure_upper = empirical_failure.saturating_add(confidence).min(900_000) as u32;
        if failure_upper > settings.maximum_failure_upper_ppm {
            return None;
        }
        let conservative_time = self
            .ratio_ppm
            .saturating_add(self.deviation_ppm.min(self.ratio_ppm));
        let retry = u64::from(MULTIPLIER_ONE)
            .saturating_mul(u64::from(MULTIPLIER_ONE))
            .checked_div(u64::from(MULTIPLIER_ONE.saturating_sub(failure_upper)).max(1))
            .unwrap_or(u64::MAX);
        let multiplier = u64::from(conservative_time)
            .saturating_mul(retry)
            .checked_div(u64::from(MULTIPLIER_ONE))
            .unwrap_or(u64::MAX);
        Some(u32::try_from(multiplier).unwrap_or(u32::MAX).clamp(
            settings.minimum_multiplier_ppm,
            settings.maximum_multiplier_ppm,
        ))
    }

    fn validate(&self) -> bool {
        u64::from(self.successes) + u64::from(self.failures) + u64::from(self.interruptions)
            == u64::from(self.observations)
            && self.observations > 0
            && self.observations <= u32::MAX / 2
            && self.ratio_ppm <= MAX_SAMPLE_MULTIPLIER
            && self.deviation_ppm <= MAX_SAMPLE_MULTIPLIER
            && (self.successes == 0) == (self.ratio_ppm == 0)
    }

    fn can_observe(&self) -> bool {
        self.observations < u32::MAX / 2
    }
}

fn integer_sqrt(value: u64) -> u64 {
    if value < 2 {
        return value;
    }
    let mut x = value;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + value / x) / 2;
    }
    x
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveMode {
    Disabled,
    #[default]
    Shadow,
    Enabled,
}

#[derive(Debug, Clone)]
pub struct AdaptiveSettings {
    pub mode: AdaptiveMode,
    pub min_samples: u32,
    pub max_buckets: usize,
    /// Maximum relative change introduced by one promotion.
    pub max_step_percent: u32,
    pub minimum_multiplier_ppm: u32,
    pub maximum_multiplier_ppm: u32,
    pub maximum_failure_upper_ppm: u32,
    pub minimum_evaluation_journeys: u32,
    pub maximum_p95_regression_percent: u32,
}

impl Default for AdaptiveSettings {
    fn default() -> Self {
        Self {
            mode: AdaptiveMode::Shadow,
            min_samples: DEFAULT_MIN_SAMPLES,
            max_buckets: MAX_PROFILE_BUCKETS,
            max_step_percent: 10,
            minimum_multiplier_ppm: 500_000,
            maximum_multiplier_ppm: 2_000_000,
            maximum_failure_upper_ppm: 350_000,
            minimum_evaluation_journeys: 20,
            maximum_p95_regression_percent: 2,
        }
    }
}

impl AdaptiveSettings {
    fn validated(&self) -> Self {
        let minimum = self.minimum_multiplier_ppm.clamp(100_000, MULTIPLIER_ONE);
        Self {
            mode: self.mode,
            min_samples: self.min_samples.max(3),
            max_buckets: self.max_buckets.clamp(1, MAX_PROFILE_BUCKETS),
            max_step_percent: self.max_step_percent.clamp(1, 25),
            minimum_multiplier_ppm: minimum,
            maximum_multiplier_ppm: self
                .maximum_multiplier_ppm
                .clamp(MULTIPLIER_ONE, 4_000_000)
                .max(minimum),
            maximum_failure_upper_ppm: self.maximum_failure_upper_ppm.min(900_000),
            minimum_evaluation_journeys: self.minimum_evaluation_journeys.max(1),
            maximum_p95_regression_percent: self.maximum_p95_regression_percent.min(10),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct PromotedModel {
    revision: u64,
    source_data_revision: u64,
    multipliers: BTreeMap<FeatureKey, u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveRegime {
    pub build_id: String,
    pub planner_settings_hash: u64,
    pub control_settings_hash: u64,
    pub baseline_costs_hash: u64,
    pub actor_capability_hash: u64,
}

impl AdaptiveRegime {
    pub fn validate_for(&self, key: &ProfileKey) -> bool {
        valid_context_text(&self.build_id)
            && self.planner_settings_hash != 0
            && self.control_settings_hash != 0
            && self.baseline_costs_hash != 0
            && self.actor_capability_hash == key.capability_hash
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveProfile {
    schema: u16,
    pub key: ProfileKey,
    /// Accepted telemetry revision.
    pub data_revision: u64,
    regime: Option<AdaptiveRegime>,
    estimates: BTreeMap<FeatureKey, PrimitiveEstimate>,
    active: PromotedModel,
    previous: Option<PromotedModel>,
    recent_completions: BTreeMap<u64, ObservationId>,
    replay_high_water: u64,
    last_evaluation_id: u64,
}

impl AdaptiveProfile {
    pub fn new(key: ProfileKey) -> Self {
        Self {
            schema: ADAPTIVE_SCHEMA_VERSION,
            key,
            data_revision: 0,
            regime: None,
            estimates: BTreeMap::new(),
            active: PromotedModel::default(),
            previous: None,
            recent_completions: BTreeMap::new(),
            replay_high_water: 0,
            last_evaluation_id: 0,
        }
    }

    pub fn model_revision(&self) -> u64 {
        self.active.revision
    }

    pub(crate) fn prepare_nonce_allocator(&self) -> Result<(), ProfileError> {
        ensure_next_observation_nonce_after(self.replay_high_water)
    }

    /// Records one terminal observation. Replays and duplicate completions are
    /// rejected before they can affect any estimate.
    pub fn observe(&mut self, observation: &MoveObservation, settings: &AdaptiveSettings) -> bool {
        let settings = settings.validated();
        let Some(arrival_sequence) = self.data_revision.checked_add(1) else {
            return false;
        };
        let replay_floor = self
            .replay_high_water
            .saturating_sub(MAX_RECENT_COMPLETIONS as u64);
        if !observation.validate()
            || observation.context.profile != self.key
            || self
                .regime
                .as_ref()
                .is_some_and(|regime| *regime != observation.context.regime())
            || observation.id.nonce <= replay_floor
            || self.recent_completions.contains_key(&observation.id.nonce)
        {
            return false;
        }
        if self.regime.is_none() {
            self.regime = Some(observation.context.regime());
        }

        let fallback = FeatureKey::fallback(observation.primitive);
        let exact = FeatureKey::exact(observation.primitive, observation.features);
        if self
            .estimates
            .get(&fallback)
            .is_some_and(|estimate| !estimate.can_observe())
            || self
                .estimates
                .get(&exact)
                .is_some_and(|estimate| !estimate.can_observe())
        {
            return false;
        }
        if !self.estimates.contains_key(&exact)
            && self.estimates.keys().filter(|key| key.is_exact()).count() >= settings.max_buckets
        {
            self.evict_least_useful_exact();
        }
        self.estimates
            .entry(fallback)
            .or_default()
            .observe(observation, arrival_sequence);
        self.estimates
            .entry(exact)
            .or_default()
            .observe(observation, arrival_sequence);
        self.data_revision = arrival_sequence;
        self.recent_completions
            .insert(observation.id.nonce, observation.id);
        self.replay_high_water = self.replay_high_water.max(observation.id.nonce);
        let replay_floor = self
            .replay_high_water
            .saturating_sub(MAX_RECENT_COMPLETIONS as u64);
        while self
            .recent_completions
            .first_key_value()
            .is_some_and(|(&nonce, _)| nonce <= replay_floor)
        {
            self.recent_completions.pop_first();
        }
        true
    }

    fn evict_least_useful_exact(&mut self) {
        let victim = self
            .estimates
            .iter()
            .filter(|(key, _)| key.is_exact())
            .min_by_key(|(key, estimate)| (estimate.trials(), estimate.last_sequence, **key))
            .map(|(key, _)| *key);
        if let Some(victim) = victim {
            self.estimates.remove(&victim);
        }
    }

    fn candidate_multipliers(&self, settings: &AdaptiveSettings) -> BTreeMap<FeatureKey, u32> {
        let settings = settings.validated();
        self.estimates
            .iter()
            .filter_map(|(&key, estimate)| {
                estimate
                    .proposed_multiplier(&settings)
                    .map(|multiplier| (key, multiplier))
            })
            .collect()
    }

    fn bounded_candidate(&self, settings: &AdaptiveSettings) -> BTreeMap<FeatureKey, u32> {
        let settings = settings.validated();
        self.candidate_multipliers(&settings)
            .into_iter()
            .map(|(key, proposed)| {
                let current = self
                    .active
                    .multipliers
                    .get(&key)
                    .copied()
                    .unwrap_or(MULTIPLIER_ONE);
                let delta =
                    (u64::from(current) * u64::from(settings.max_step_percent) / 100).max(1) as u32;
                let bounded = proposed
                    .clamp(current.saturating_sub(delta), current.saturating_add(delta))
                    .clamp(
                        settings.minimum_multiplier_ppm,
                        settings.maximum_multiplier_ppm,
                    );
                (key, bounded)
            })
            .collect()
    }

    pub fn candidate_hash(&self, settings: &AdaptiveSettings) -> u64 {
        let settings = settings.validated();
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"azalea-pathfinder/candidate-model/v2");
        hasher.update(&self.schema.to_le_bytes());
        hasher.update(&self.data_revision.to_le_bytes());
        hasher.update(&self.active.revision.to_le_bytes());
        hasher.update(&[match settings.mode {
            AdaptiveMode::Disabled => 0,
            AdaptiveMode::Shadow => 1,
            AdaptiveMode::Enabled => 2,
        }]);
        hasher.update(&settings.min_samples.to_le_bytes());
        hasher.update(&(settings.max_buckets as u64).to_le_bytes());
        hasher.update(&settings.max_step_percent.to_le_bytes());
        hasher.update(&settings.minimum_multiplier_ppm.to_le_bytes());
        hasher.update(&settings.maximum_multiplier_ppm.to_le_bytes());
        hasher.update(&settings.maximum_failure_upper_ppm.to_le_bytes());
        hasher.update(&settings.minimum_evaluation_journeys.to_le_bytes());
        hasher.update(&settings.maximum_p95_regression_percent.to_le_bytes());
        if let Some(regime) = &self.regime {
            hasher.update(&[1]);
            hash_text(&mut hasher, &regime.build_id);
            hasher.update(&regime.planner_settings_hash.to_le_bytes());
            hasher.update(&regime.control_settings_hash.to_le_bytes());
            hasher.update(&regime.baseline_costs_hash.to_le_bytes());
            hasher.update(&regime.actor_capability_hash.to_le_bytes());
        } else {
            hasher.update(&[0]);
        }
        for (key, multiplier) in self.bounded_candidate(&settings) {
            hasher.update(&key.0.to_le_bytes());
            hasher.update(&multiplier.to_le_bytes());
        }
        u64::from_le_bytes(
            hasher.finalize().as_bytes()[..8]
                .try_into()
                .expect("eight digest bytes"),
        )
    }

    pub fn snapshot(
        &self,
        baseline: MovementCosts,
        settings: &AdaptiveSettings,
        regime: &AdaptiveRegime,
    ) -> CostModelSnapshot {
        let regime_matches = regime.validate_for(&self.key)
            && movement_costs_hash(baseline) == regime.baseline_costs_hash
            && self.regime.as_ref() == Some(regime);
        CostModelSnapshot {
            data_revision: if regime_matches {
                self.data_revision
            } else {
                0
            },
            model_revision: if regime_matches {
                self.active.revision
            } else {
                0
            },
            mode: settings.validated().mode,
            baseline,
            active: if regime_matches {
                self.active.multipliers.clone()
            } else {
                BTreeMap::new()
            },
            candidate: if regime_matches {
                self.bounded_candidate(settings)
            } else {
                BTreeMap::new()
            },
        }
    }

    /// Promotes a shadow candidate only when its paired evaluation is at least
    /// as safe as the known-good baseline and does not materially regress p95.
    pub fn promote(
        &mut self,
        report: &PromotionReport,
        settings: &AdaptiveSettings,
    ) -> Result<u64, PromotionError> {
        let settings = settings.validated();
        let bounded = self.bounded_candidate(&settings);
        if report.candidate_data_revision != self.data_revision
            || report.expected_model_revision != self.active.revision
            || report.candidate_hash != self.candidate_hash(&settings)
            || report.evaluation_id == 0
            || report.evaluation_id <= self.last_evaluation_id
        {
            return Err(PromotionError::StaleCandidate);
        }
        report.validate(&settings)?;
        if bounded.is_empty() {
            return Err(PromotionError::InsufficientSamples);
        }
        let next_revision = self
            .active
            .revision
            .checked_add(1)
            .ok_or(PromotionError::RevisionOverflow)?;
        self.previous = Some(self.active.clone());
        self.active = PromotedModel {
            revision: next_revision,
            source_data_revision: self.data_revision,
            multipliers: bounded,
        };
        self.last_evaluation_id = report.evaluation_id;
        Ok(self.active.revision)
    }

    pub fn rollback(&mut self, _signal: GuardSignal) -> Result<u64, PromotionError> {
        let Some(mut previous) = self.previous.take() else {
            return Err(PromotionError::NoRollback);
        };
        previous.revision = self
            .active
            .revision
            .checked_add(1)
            .ok_or(PromotionError::RevisionOverflow)?;
        self.active = previous;
        Ok(self.active.revision)
    }

    pub fn validate(&self) -> bool {
        let exact_buckets = self.estimates.keys().filter(|key| key.is_exact()).count();
        let fallback_buckets = self.estimates.len().saturating_sub(exact_buckets);
        let completions_valid = self.recent_completions.iter().all(|(&nonce, completion)| {
            nonce != 0
                && nonce == completion.nonce
                && nonce <= self.replay_high_water
                && nonce
                    > self
                        .replay_high_water
                        .saturating_sub(MAX_RECENT_COMPLETIONS as u64)
        });
        let active_valid = self.active.source_data_revision <= self.data_revision
            && (self.active.revision != 0
                || (self.active.source_data_revision == 0 && self.active.multipliers.is_empty()));
        let previous_valid = self.previous.as_ref().is_none_or(|previous| {
            previous.revision < self.active.revision
                && previous.source_data_revision <= self.data_revision
        });
        let pristine_or_initialized = if self.data_revision == 0 {
            self.regime.is_none()
                && self.estimates.is_empty()
                && self.recent_completions.is_empty()
                && self.replay_high_water == 0
                && self.active == PromotedModel::default()
                && self.previous.is_none()
                && self.last_evaluation_id == 0
        } else {
            self.regime.is_some()
                && !self.estimates.is_empty()
                && !self.recent_completions.is_empty()
                && self.replay_high_water != 0
                && self.replay_high_water < MAX_OBSERVATION_NONCE
        };
        self.schema == ADAPTIVE_SCHEMA_VERSION
            && self.key.validate()
            && exact_buckets <= MAX_PROFILE_BUCKETS
            && fallback_buckets <= 11
            && self.active.multipliers.len() <= MAX_PROFILE_BUCKETS + 11
            && self
                .previous
                .as_ref()
                .is_none_or(|model| model.multipliers.len() <= MAX_PROFILE_BUCKETS + 11)
            && self.estimates.keys().all(|key| key.validate())
            && self.estimates.values().all(PrimitiveEstimate::validate)
            && self.estimates.values().all(|estimate| {
                estimate.last_sequence != 0 && estimate.last_sequence <= self.data_revision
            })
            && self.recent_completions.len() <= MAX_RECENT_COMPLETIONS
            && self
                .regime
                .as_ref()
                .is_none_or(|regime| regime.validate_for(&self.key))
            && completions_valid
            && self
                .active
                .multipliers
                .keys()
                .chain(
                    self.previous
                        .iter()
                        .flat_map(|model| model.multipliers.keys()),
                )
                .all(|key| key.validate())
            && self
                .active
                .multipliers
                .values()
                .chain(
                    self.previous
                        .iter()
                        .flat_map(|model| model.multipliers.values()),
                )
                .all(|value| (100_000..=4_000_000).contains(value))
            && active_valid
            && previous_valid
            && pristine_or_initialized
            && (self.active.revision == 0 || self.last_evaluation_id != 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardSignal {
    Disconnect,
    DamageSpike,
    SetbackSpike,
    CorrectionSpike,
    Operator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JourneyMetrics {
    pub journeys: u32,
    pub failed: u32,
    pub p95_ticks: u32,
    pub damage_half_hearts: u32,
    pub setbacks: u32,
    pub corrections: u32,
    pub disconnects: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromotionReport {
    pub evaluation_id: u64,
    pub candidate_data_revision: u64,
    pub expected_model_revision: u64,
    pub candidate_hash: u64,
    pub known_good: JourneyMetrics,
    pub shadow: JourneyMetrics,
}

impl PromotionReport {
    fn validate(&self, settings: &AdaptiveSettings) -> Result<(), PromotionError> {
        if self.known_good.journeys < settings.minimum_evaluation_journeys
            || self.shadow.journeys < settings.minimum_evaluation_journeys
        {
            return Err(PromotionError::InsufficientEvaluation);
        }
        if self.known_good.failed > self.known_good.journeys
            || self.shadow.failed > self.shadow.journeys
            || self.known_good.p95_ticks == 0
            || self.shadow.p95_ticks == 0
        {
            return Err(PromotionError::InvalidReport);
        }
        if rate_worse(
            self.shadow.failed,
            self.shadow.journeys,
            self.known_good.failed,
            self.known_good.journeys,
        ) || rate_worse(
            self.shadow.damage_half_hearts,
            self.shadow.journeys,
            self.known_good.damage_half_hearts,
            self.known_good.journeys,
        ) || rate_worse(
            self.shadow.setbacks,
            self.shadow.journeys,
            self.known_good.setbacks,
            self.known_good.journeys,
        ) || rate_worse(
            self.shadow.corrections,
            self.shadow.journeys,
            self.known_good.corrections,
            self.known_good.journeys,
        ) || rate_worse(
            self.shadow.disconnects,
            self.shadow.journeys,
            self.known_good.disconnects,
            self.known_good.journeys,
        ) {
            return Err(PromotionError::SafetyRegression);
        }
        let allowed_p95 = u64::from(self.known_good.p95_ticks)
            .saturating_mul(u64::from(100 + settings.maximum_p95_regression_percent))
            / 100;
        if u64::from(self.shadow.p95_ticks) > allowed_p95 {
            return Err(PromotionError::PerformanceRegression);
        }
        Ok(())
    }
}

fn rate_worse(candidate_bad: u32, candidate_total: u32, base_bad: u32, base_total: u32) -> bool {
    u64::from(candidate_bad).saturating_mul(u64::from(base_total))
        > u64::from(base_bad).saturating_mul(u64::from(candidate_total))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionError {
    ProfileCapacity,
    RevisionOverflow,
    StaleCandidate,
    InsufficientSamples,
    InsufficientEvaluation,
    InvalidReport,
    SafetyRegression,
    PerformanceRegression,
    NoRollback,
}

impl std::fmt::Display for PromotionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "adaptive promotion rejected: {self:?}")
    }
}

impl std::error::Error for PromotionError {}

#[derive(Debug, Clone)]
pub struct CostModelSnapshot {
    pub data_revision: u64,
    pub model_revision: u64,
    pub mode: AdaptiveMode,
    baseline: MovementCosts,
    active: BTreeMap<FeatureKey, u32>,
    candidate: BTreeMap<FeatureKey, u32>,
}

impl CostModelSnapshot {
    /// Whether this snapshot can change route ordering. Shadow, disabled, and
    /// regime-mismatched snapshots are exact baseline models.
    pub fn affects_routing(&self) -> bool {
        self.mode == AdaptiveMode::Enabled && !self.active.is_empty()
    }

    /// Candidate scalar helper for costs that contain only adaptable time and
    /// retry work. Use [`Self::edge_components`] whenever an edge also carries
    /// immutable safety, damage, desync, or resource costs.
    pub fn proposed_edge_cost(
        &self,
        primitive: PrimitiveId,
        features: MoveFeatures,
        baseline_edge_cost: Cost,
    ) -> Option<Cost> {
        let multiplier = lookup_multiplier(&self.candidate, primitive, features)?;
        Some(scale_cost(baseline_edge_cost, multiplier))
    }

    /// Applies the exact step-bounded shadow candidate to adaptable
    /// components while preserving all fixed safety components.
    pub fn proposed_edge_components(
        &self,
        primitive: PrimitiveId,
        features: MoveFeatures,
        mut baseline: CostComponents,
    ) -> Option<CostComponents> {
        let multiplier = lookup_multiplier(&self.candidate, primitive, features)?;
        baseline.time = scale_cost_preserving_zero(baseline.time, multiplier);
        baseline.reliability = scale_cost_preserving_zero(baseline.reliability, multiplier);
        Some(baseline)
    }

    /// Produces an offline routing view of the exact candidate bound into
    /// [`AdaptiveProfile::candidate_hash`]. It does not mutate or promote the
    /// profile and is intended for paired shadow evaluation.
    pub fn candidate_view(&self) -> Self {
        Self {
            data_revision: self.data_revision,
            model_revision: self.model_revision,
            mode: AdaptiveMode::Enabled,
            baseline: self.baseline,
            active: self.candidate.clone(),
            candidate: self.candidate.clone(),
        }
    }

    /// Returns an active-model scalar cost containing only adaptable work.
    /// Shadow and disabled snapshots leave the baseline unchanged. Use
    /// [`Self::edge_components`] for mixed adaptable/immutable costs.
    pub fn edge_cost(
        &self,
        primitive: PrimitiveId,
        features: MoveFeatures,
        baseline_edge_cost: Cost,
    ) -> Cost {
        if self.mode != AdaptiveMode::Enabled {
            return baseline_edge_cost;
        }
        lookup_multiplier(&self.active, primitive, features)
            .map_or(baseline_edge_cost, |multiplier| {
                scale_cost(baseline_edge_cost, multiplier)
            })
    }

    /// Scales only the adaptable time/retry portion of an edge. Damage,
    /// desynchronization and resource/safety penalties remain unchanged even
    /// when a learned multiplier is below one.
    pub fn edge_components(
        &self,
        primitive: PrimitiveId,
        features: MoveFeatures,
        mut baseline: CostComponents,
    ) -> CostComponents {
        if self.mode != AdaptiveMode::Enabled {
            return baseline;
        }
        if let Some(multiplier) = lookup_multiplier(&self.active, primitive, features) {
            baseline.time = scale_cost_preserving_zero(baseline.time, multiplier);
            baseline.reliability = scale_cost_preserving_zero(baseline.reliability, multiplier);
        }
        baseline
    }

    /// Compatibility helper for one-block scalar cost fields. Fall is omitted:
    /// its fixed base and per-block slope must be scaled together with
    /// [`Self::edge_cost`], not fitted into `fall_per_block`.
    pub fn apply(&self, mut baseline: MovementCosts) -> MovementCosts {
        if self.mode != AdaptiveMode::Enabled {
            return baseline;
        }
        for (target, primitive) in [
            (&mut baseline.cardinal_walk, PrimitiveId::WALK_CARDINAL),
            (&mut baseline.diagonal_walk, PrimitiveId::WALK_DIAGONAL),
            (&mut baseline.step, PrimitiveId::STEP),
            (&mut baseline.jump, PrimitiveId::JUMP),
            (&mut baseline.parkour_per_block, PrimitiveId::PARKOUR),
            (&mut baseline.climb, PrimitiveId::CLIMB),
            (&mut baseline.swim, PrimitiveId::SWIM),
            (&mut baseline.swim_exit, PrimitiveId::SWIM_EXIT),
        ] {
            if let Some(multiplier) = self.active.get(&FeatureKey::fallback(primitive)).copied() {
                *target = scale_cost(*target, multiplier);
            }
        }
        baseline
    }

    pub fn baseline(&self) -> MovementCosts {
        self.baseline
    }
}

fn lookup_multiplier(
    model: &BTreeMap<FeatureKey, u32>,
    primitive: PrimitiveId,
    features: MoveFeatures,
) -> Option<u32> {
    model
        .get(&FeatureKey::exact(primitive, features))
        .or_else(|| model.get(&FeatureKey::fallback(primitive)))
        .copied()
}

fn scale_cost(cost: Cost, multiplier: u32) -> Cost {
    scale_cost_preserving_zero(cost, multiplier).max(1)
}

fn scale_cost_preserving_zero(cost: Cost, multiplier: u32) -> Cost {
    if cost == 0 {
        return 0;
    }
    let scaled = u64::from(cost)
        .saturating_mul(u64::from(multiplier))
        .saturating_add(u64::from(MULTIPLIER_ONE - 1))
        / u64::from(MULTIPLIER_ONE);
    u32::try_from(scaled).unwrap_or(u32::MAX)
}

pub fn movement_costs_hash(costs: MovementCosts) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"azalea-pathfinder/movement-costs/v1");
    for value in [
        costs.cardinal_walk,
        costs.diagonal_walk,
        costs.step,
        costs.jump,
        costs.fall_base,
        costs.fall_per_block,
        costs.parkour_per_block,
        costs.swim,
        costs.climb,
        costs.swim_exit,
    ] {
        hasher.update(&value.to_le_bytes());
    }
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("eight digest bytes"),
    )
}

/// Stable hash of planner policy frozen for a leg. Journey-local avoidance
/// entries and the randomized tie-break seed are intentionally excluded; they
/// are route state rather than model semantics.
pub fn move_context_hash(context: &MoveContext) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"azalea-pathfinder/move-context/v1");
    hasher.update(&movement_costs_hash(context.costs).to_le_bytes());
    for value in [
        context.max_fall as i64,
        context.goal_tolerance as i64,
        context.max_expansions.min(u64::MAX as usize) as i64,
        context.time_budget_ms.min(i64::MAX as u64) as i64,
        i64::from(context.fall_damage_penalty),
        i64::from(context.wall_penalty),
        i64::from(context.lava_penalty),
        i64::from(context.lava_proximity_radius),
        i64::from(context.lava_proximity_penalty),
        i64::from(context.water_penalty),
        i64::from(context.grazing_step_penalty),
    ] {
        hasher.update(&value.to_le_bytes());
    }
    match context.lava_policy {
        LavaPolicy::Forbidden { clearance } => {
            hasher.update(&[0]);
            hasher.update(&clearance.to_le_bytes());
        }
        LavaPolicy::Penalized => {
            hasher.update(&[1]);
        }
    }
    hasher.update(&[match context.water_policy {
        WaterPolicy::Forbidden => 0,
        WaterPolicy::Penalized => 1,
    }]);
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("eight digest bytes"),
    )
}

/// Stable hash of the execution controls that determine how planned
/// primitives are actually driven. This is kept separate from planner policy
/// so telemetry gathered with different steering, timeout, or retry behavior
/// cannot be pooled accidentally.
pub fn follower_settings_hash(settings: &FollowerSettings) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"azalea-pathfinder/follower-settings/v1");
    for value in [
        settings.node_radius_xz,
        settings.node_y_tolerance,
        settings.passed_node_radius_xz,
        settings.arrival_radius_xz,
        settings.arrival_y_tolerance,
        settings.close_enough_xz,
        settings.close_enough_y,
        settings.line_sample_spacing,
        settings.body_half_width,
        settings.jump_height_threshold,
        settings.minimum_lookahead,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    for value in [
        settings.minimum_turn_degrees,
        settings.yaw_wander_speed,
        settings.yaw_wander_degrees,
        settings.pitch_wander_speed,
        settings.pitch_wander_degrees,
        settings.turn_ease,
        settings.pitch_clamp_degrees,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    for value in [
        u64::from(settings.stall_ticks),
        u64::from(settings.max_follow_ticks),
        settings.max_los_skip.min(u64::MAX as usize) as u64,
        settings.turn_jitter_degrees,
        settings.sprint_break_chance_denominator,
        u64::from(settings.sprint_break_min_ticks),
        u64::from(settings.sprint_break_max_ticks),
        u64::from(settings.escape_hop_first_min_ticks),
        u64::from(settings.escape_hop_first_max_ticks),
        u64::from(settings.escape_hop_second_min_ticks),
        u64::from(settings.escape_hop_second_max_ticks),
    ] {
        hasher.update(&value.to_le_bytes());
    }
    match settings.lava_policy {
        LavaPolicy::Forbidden { clearance } => {
            hasher.update(&[0]);
            hasher.update(&clearance.to_le_bytes());
        }
        LavaPolicy::Penalized => {
            hasher.update(&[1]);
        }
    }
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("eight digest bytes"),
    )
}

#[derive(Debug)]
pub enum ProfileError {
    Io(std::io::Error),
    Invalid,
    TooLarge,
    Capacity,
    AlreadyExists,
    NonceExhausted,
    Json(serde_json::Error),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "profile I/O failed: {error}"),
            Self::Invalid => write!(
                formatter,
                "profile is invalid or belongs to another context"
            ),
            Self::TooLarge => write!(formatter, "profile exceeds the size limit"),
            Self::Capacity => write!(formatter, "adaptive profile capacity is exhausted"),
            Self::AlreadyExists => write!(formatter, "adaptive profile already exists"),
            Self::NonceExhausted => write!(formatter, "observation nonce space is exhausted"),
            Self::Json(error) => write!(formatter, "profile JSON is invalid: {error}"),
        }
    }
}

impl std::error::Error for ProfileError {}

impl From<std::io::Error> for ProfileError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for ProfileError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

fn profile_write_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

pub fn profile_path(root: &Path, key: &ProfileKey) -> PathBuf {
    root.join(format!("profile-{}.json", key.digest().to_hex()))
}

pub fn load_profile(root: &Path, key: &ProfileKey) -> Result<AdaptiveProfile, ProfileError> {
    let profile = load_profile_unprepared(root, key)?;
    profile.prepare_nonce_allocator()?;
    Ok(profile)
}

pub(crate) fn load_profile_unprepared(
    root: &Path,
    key: &ProfileKey,
) -> Result<AdaptiveProfile, ProfileError> {
    let mut file = File::open(profile_path(root, key))?;
    if file.metadata()?.len() > MAX_PROFILE_BYTES {
        return Err(ProfileError::TooLarge);
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(MAX_PROFILE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_PROFILE_BYTES {
        return Err(ProfileError::TooLarge);
    }
    let profile: AdaptiveProfile = serde_json::from_slice(&bytes)?;
    if !profile.validate() || &profile.key != key {
        return Err(ProfileError::Invalid);
    }
    Ok(profile)
}

pub fn save_profile_atomic(root: &Path, profile: &AdaptiveProfile) -> Result<(), ProfileError> {
    if !profile.validate() {
        return Err(ProfileError::Invalid);
    }
    let encoded = serde_json::to_vec(profile)?;
    if encoded.len() as u64 > MAX_PROFILE_BYTES {
        return Err(ProfileError::TooLarge);
    }
    let _guard = profile_write_lock()
        .lock()
        .map_err(|_| ProfileError::Invalid)?;
    fs::create_dir_all(root)?;
    let mut file = AtomicWriteFile::open(profile_path(root, &profile.key))?;
    file.write_all(&encoded)?;
    file.flush()?;
    file.sync_all()?;
    file.commit()?;
    Ok(())
}

/// Non-blocking producer for the optional JSON-lines telemetry worker.
#[derive(Clone)]
pub struct TelemetryQueue {
    sender: SyncSender<MoveObservation>,
    dropped: Arc<AtomicU64>,
    gate: Arc<TelemetryGate>,
}

struct TelemetryGate {
    closing: AtomicBool,
    in_flight: AtomicUsize,
}

impl TelemetryQueue {
    pub fn try_send(&self, observation: MoveObservation) -> bool {
        if self.gate.closing.load(Ordering::Acquire) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        self.gate.in_flight.fetch_add(1, Ordering::AcqRel);
        if self.gate.closing.load(Ordering::Acquire) {
            self.gate.in_flight.fetch_sub(1, Ordering::Release);
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let sent = match self.sender.try_send(observation) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        };
        self.gate.in_flight.fetch_sub(1, Ordering::Release);
        sent
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

enum WorkerControl {
    Shutdown(Option<SyncSender<()>>),
}

fn telemetry_registry() -> &'static Mutex<HashSet<PathBuf>> {
    static REGISTRY: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashSet::new()))
}

struct WriterRegistration(PathBuf);

impl Drop for WriterRegistration {
    fn drop(&mut self) {
        if let Ok(mut registry) = telemetry_registry().lock() {
            registry.remove(&self.0);
        }
    }
}

/// Optional disk sink. Spawning a second writer for the same profile is
/// rejected, and all file I/O stays on the worker thread.
pub struct TelemetryWorker {
    queue: Option<TelemetryQueue>,
    gate: Arc<TelemetryGate>,
    control: Option<mpsc::Sender<WorkerControl>>,
    handle: Option<JoinHandle<()>>,
}

impl TelemetryWorker {
    pub fn spawn(root: PathBuf, key: ProfileKey, capacity: usize) -> std::io::Result<Self> {
        if !key.validate() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid telemetry profile",
            ));
        }
        fs::create_dir_all(&root)?;
        let root = fs::canonicalize(root)?;
        // Multiple processes use distinct streams so append and rotation never
        // race across OS processes. The in-process registry still enforces one
        // writer per profile stream.
        let path = root.join(format!(
            "telemetry-{}-{}.jsonl",
            key.digest().to_hex(),
            std::process::id()
        ));
        {
            let mut registry = telemetry_registry()
                .lock()
                .map_err(|_| std::io::Error::other("telemetry registry poisoned"))?;
            if !registry.insert(path.clone()) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "a telemetry writer already owns this profile",
                ));
            }
        }

        let (sender, receiver) = mpsc::sync_channel(capacity.clamp(1, 65_536));
        let (control_sender, control_receiver) = mpsc::channel();
        let dropped = Arc::new(AtomicU64::new(0));
        let gate = Arc::new(TelemetryGate {
            closing: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
        });
        let queue = TelemetryQueue {
            sender,
            dropped: dropped.clone(),
            gate: gate.clone(),
        };
        let worker_gate = gate.clone();
        let worker_path = path.clone();
        let spawn_result = std::thread::Builder::new()
            .name("azalea-pathfinder-telemetry".into())
            .spawn(move || {
                let _registration = WriterRegistration(worker_path.clone());
                loop {
                    match control_receiver.try_recv() {
                        Ok(WorkerControl::Shutdown(acknowledge)) => {
                            while worker_gate.in_flight.load(Ordering::Acquire) != 0 {
                                std::thread::yield_now();
                            }
                            while let Ok(observation) = receiver.try_recv() {
                                write_telemetry(&worker_path, &key, &dropped, observation);
                            }
                            if let Some(acknowledge) = acknowledge {
                                let _ = acknowledge.try_send(());
                            }
                            break;
                        }
                        Err(TryRecvError::Disconnected) => break,
                        Err(TryRecvError::Empty) => {}
                    }
                    match receiver.recv_timeout(Duration::from_millis(50)) {
                        Ok(observation) => {
                            write_telemetry(&worker_path, &key, &dropped, observation)
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
            });
        let handle = match spawn_result {
            Ok(handle) => handle,
            Err(error) => {
                if let Ok(mut registry) = telemetry_registry().lock() {
                    registry.remove(&path);
                }
                return Err(error);
            }
        };
        Ok(Self {
            queue: Some(queue),
            gate,
            control: Some(control_sender),
            handle: Some(handle),
        })
    }

    pub fn queue(&self) -> TelemetryQueue {
        self.queue
            .as_ref()
            .expect("telemetry worker queue is available until shutdown")
            .clone()
    }

    /// Drains and joins the worker. Call this from a shutdown/maintenance
    /// thread, never from `GameTick`.
    pub fn shutdown_blocking(mut self) {
        self.gate.closing.store(true, Ordering::Release);
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut acknowledged = false;
        if let Some(control) = self.control.take() {
            let _ = control.send(WorkerControl::Shutdown(Some(sender)));
            acknowledged = receiver.recv_timeout(Duration::from_secs(5)).is_ok();
        }
        self.queue.take();
        if acknowledged && let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        // On a wedged filesystem, detach instead of turning bounded shutdown
        // into an unbounded join.
        self.handle.take();
    }
}

impl Drop for TelemetryWorker {
    fn drop(&mut self) {
        // Signal a bounded drain, but deliberately do not join here: Drop may
        // run on the game schedule and must not block it on filesystem I/O.
        self.gate.closing.store(true, Ordering::Release);
        if let Some(control) = self.control.take() {
            let _ = control.send(WorkerControl::Shutdown(None));
        }
        self.queue.take();
        self.handle.take();
    }
}

fn write_telemetry(
    path: &Path,
    key: &ProfileKey,
    dropped: &AtomicU64,
    observation: MoveObservation,
) {
    if !observation.validate() || &observation.context.profile != key {
        dropped.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if fs::metadata(path).is_ok_and(|metadata| metadata.len() >= MAX_TELEMETRY_FILE_BYTES) {
        let previous = path.with_extension("previous.jsonl");
        let _ = fs::remove_file(&previous);
        if fs::rename(path, previous).is_err() {
            dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
    }
    let Ok(file) = OpenOptions::new().create(true).append(true).open(path) else {
        dropped.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let mut writer = BufWriter::new(file);
    if serde_json::to_writer(&mut writer, &observation).is_err()
        || writer.write_all(b"\n").is_err()
        || writer.flush().is_err()
    {
        dropped.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(key: &ProfileKey) -> ObservationContext {
        ObservationContext {
            profile: key.clone(),
            journey_id: 7,
            generation: 3,
            leg: 2,
            plan_revision: 11,
            world_revision: 12,
            model_revision: 0,
            planner_settings_hash: 13,
            control_settings_hash: 14,
            baseline_costs_hash: movement_costs_hash(MovementCosts::default()),
            actor_capability_hash: key.capability_hash,
            build_id: "test-build".into(),
            created_unix_ms: 15,
        }
    }

    fn observation(
        key: &ProfileKey,
        edge_index: u32,
        ticks: u64,
        outcome: ObservationOutcome,
    ) -> MoveObservation {
        let evidence = match outcome {
            ObservationOutcome::Success => AttributionEvidence::ReachedPlannedNode,
            ObservationOutcome::Stalled => AttributionEvidence::StallDetector,
            ObservationOutcome::Unsafe => AttributionEvidence::LiveValidationFailure,
            ObservationOutcome::FellOff => AttributionEvidence::FellBelowPath,
            ObservationOutcome::TimedOut => AttributionEvidence::CumulativeDeadline,
            ObservationOutcome::Interrupted(
                InterruptionReason::Pause
                | InterruptionReason::Cancelled
                | InterruptionReason::GoalChanged,
            ) => AttributionEvidence::PauseOrCancellation,
            ObservationOutcome::Interrupted(_) => AttributionEvidence::ExternalMovementEvent,
        };
        MoveAttempt::begin(
            context(key),
            ObservationId {
                nonce: next_observation_nonce(),
                journey_id: 7,
                generation: 3,
                leg: 2,
                edge_index,
                attempt: 0,
            },
            PrimitiveId::WALK_CARDINAL,
            MoveFeatures::new(1, 0, TerrainClass::FullBlock),
            BlockPos::new(edge_index as i32, 64, 0),
            BlockPos::new(edge_index as i32 + 1, 64, 0),
            CostComponents {
                time: 10,
                ..CostComponents::default()
            },
            5,
            100,
        )
        .unwrap()
        .finish(100 + ticks, u64::from(edge_index) + 1, outcome, evidence)
        .unwrap()
    }

    fn permissive_settings(mode: AdaptiveMode) -> AdaptiveSettings {
        AdaptiveSettings {
            mode,
            min_samples: 3,
            maximum_failure_upper_ppm: 900_000,
            minimum_evaluation_journeys: 2,
            ..AdaptiveSettings::default()
        }
    }

    fn good_report(
        profile: &AdaptiveProfile,
        settings: &AdaptiveSettings,
        evaluation_id: u64,
    ) -> PromotionReport {
        let known_good = JourneyMetrics {
            journeys: 10,
            failed: 1,
            p95_ticks: 100,
            damage_half_hearts: 2,
            setbacks: 2,
            corrections: 2,
            disconnects: 0,
        };
        PromotionReport {
            evaluation_id,
            candidate_data_revision: profile.data_revision,
            expected_model_revision: profile.model_revision(),
            candidate_hash: profile.candidate_hash(settings),
            known_good,
            shadow: JourneyMetrics {
                p95_ticks: 99,
                ..known_good
            },
        }
    }

    #[test]
    fn attempt_is_exactly_once_and_rejects_zero_tick_success() {
        assert_eq!(
            nonce_successor(MAX_OBSERVATION_NONCE - 1),
            Some(MAX_OBSERVATION_NONCE)
        );
        assert_eq!(nonce_successor(MAX_OBSERVATION_NONCE), None);
        let key = ProfileKey::local_default();
        let attempt = MoveAttempt::begin(
            context(&key),
            ObservationId {
                nonce: next_observation_nonce(),
                journey_id: 7,
                generation: 3,
                leg: 2,
                edge_index: 0,
                attempt: 0,
            },
            PrimitiveId::WALK_CARDINAL,
            MoveFeatures::new(1, 0, TerrainClass::FullBlock),
            BlockPos::new(0, 64, 0),
            BlockPos::new(1, 64, 0),
            CostComponents {
                time: 10,
                ..CostComponents::default()
            },
            5,
            100,
        )
        .unwrap();
        assert!(matches!(
            attempt.finish(
                100,
                1,
                ObservationOutcome::Success,
                AttributionEvidence::ReachedPlannedNode
            ),
            Err(ObservationError::TickOrder)
        ));
    }

    #[test]
    fn duplicate_completion_and_unknown_primitive_are_rejected() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = permissive_settings(AdaptiveMode::Shadow);
        let first_observation = observation(&key, 0, 5, ObservationOutcome::Success);
        assert!(profile.observe(&first_observation, &settings));
        assert!(!profile.observe(&first_observation, &settings));
        let mut reused_nonce = observation(&key, 1, 5, ObservationOutcome::Success);
        reused_nonce.id.nonce = first_observation.id.nonce;
        assert!(
            !profile.observe(&reused_nonce, &settings),
            "a nonce is globally unique even if other identity fields differ"
        );
        let mut exhausted_nonce = observation(&key, 2, 5, ObservationOutcome::Success);
        exhausted_nonce.id.nonce = MAX_OBSERVATION_NONCE;
        assert!(!exhausted_nonce.validate());
        assert!(!profile.observe(&exhausted_nonce, &settings));
        assert!(
            MoveAttempt::begin(
                context(&key),
                ObservationId {
                    nonce: next_observation_nonce(),
                    journey_id: 7,
                    generation: 3,
                    leg: 2,
                    edge_index: 1,
                    attempt: 0,
                },
                PrimitiveId(500),
                MoveFeatures::new(1, 0, TerrainClass::FullBlock),
                BlockPos::new(0, 64, 0),
                BlockPos::new(1, 64, 0),
                CostComponents {
                    time: 10,
                    ..CostComponents::default()
                },
                5,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn interruptions_do_not_increase_failure_trials() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = permissive_settings(AdaptiveMode::Shadow);
        assert!(profile.observe(
            &observation(&key, 0, 5, ObservationOutcome::Success),
            &settings
        ));
        assert!(profile.observe(
            &observation(
                &key,
                1,
                5,
                ObservationOutcome::Interrupted(InterruptionReason::ServerCorrection)
            ),
            &settings
        ));
        let estimate = &profile.estimates[&FeatureKey::fallback(PrimitiveId::WALK_CARDINAL)];
        assert_eq!(estimate.successes, 1);
        assert_eq!(estimate.failures, 0);
        assert_eq!(estimate.interruptions, 1);
    }

    #[test]
    fn shadow_never_changes_cost_and_enabled_requires_promotion() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = permissive_settings(AdaptiveMode::Enabled);
        for edge in 0..3 {
            profile.observe(
                &observation(&key, edge, 10, ObservationOutcome::Success),
                &settings,
            );
        }
        let features = MoveFeatures::new(1, 0, TerrainClass::FullBlock);
        let regime = context(&key).regime();
        let before = profile.snapshot(MovementCosts::default(), &settings, &regime);
        assert_eq!(
            before.edge_cost(PrimitiveId::WALK_CARDINAL, features, 10),
            10
        );
        assert!(
            before
                .proposed_edge_cost(PrimitiveId::WALK_CARDINAL, features, 10)
                .is_some()
        );
        let report = good_report(&profile, &settings, 1);
        profile.promote(&report, &settings).unwrap();
        let after = profile.snapshot(MovementCosts::default(), &settings, &regime);
        assert!(after.edge_cost(PrimitiveId::WALK_CARDINAL, features, 10) > 10);
    }

    #[test]
    fn promotion_rejects_regression_and_rolls_back() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = permissive_settings(AdaptiveMode::Enabled);
        for edge in 0..3 {
            profile.observe(
                &observation(&key, edge, 10, ObservationOutcome::Success),
                &settings,
            );
        }
        let mut bad = good_report(&profile, &settings, 1);
        bad.shadow.disconnects = 1;
        assert_eq!(
            profile.promote(&bad, &settings),
            Err(PromotionError::SafetyRegression)
        );
        let report = good_report(&profile, &settings, 1);
        assert_eq!(profile.promote(&report, &settings).unwrap(), 1);
        assert_eq!(
            profile.promote(&report, &settings),
            Err(PromotionError::StaleCandidate),
            "the same evaluation cannot advance multiple promotion steps"
        );
        assert_eq!(profile.rollback(GuardSignal::Operator).unwrap(), 2);
        assert_eq!(profile.model_revision(), 2);
    }

    #[test]
    fn context_bucket_uses_fallback_until_exact_is_confident() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = permissive_settings(AdaptiveMode::Shadow);
        for edge in 0..3 {
            profile.observe(
                &observation(&key, edge, 10, ObservationOutcome::Success),
                &settings,
            );
        }
        let regime = context(&key).regime();
        let snapshot = profile.snapshot(MovementCosts::default(), &settings, &regime);
        let unseen = MoveFeatures::new(1, 0, TerrainClass::PartialBlock);
        assert!(
            snapshot
                .proposed_edge_cost(PrimitiveId::WALK_CARDINAL, unseen, 10)
                .is_some(),
            "primitive fallback should cover sparse feature buckets"
        );
    }

    #[test]
    fn changed_regime_neither_learns_from_nor_routes_with_the_old_model() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = permissive_settings(AdaptiveMode::Enabled);
        for edge in 0..3 {
            assert!(profile.observe(
                &observation(&key, edge, 10, ObservationOutcome::Success),
                &settings,
            ));
        }
        let report = good_report(&profile, &settings, 1);
        profile.promote(&report, &settings).unwrap();

        let mut changed_context = context(&key);
        changed_context.build_id = "different-build".into();
        let changed_regime = changed_context.regime();
        let snapshot = profile.snapshot(MovementCosts::default(), &settings, &changed_regime);
        let features = MoveFeatures::new(1, 0, TerrainClass::FullBlock);
        assert_eq!(
            snapshot.edge_cost(PrimitiveId::WALK_CARDINAL, features, 10),
            10
        );
        assert_eq!(snapshot.model_revision, 0);

        let mismatched_baseline = MovementCosts {
            cardinal_walk: 99,
            ..MovementCosts::default()
        };
        let original_regime = context(&key).regime();
        let mismatched = profile.snapshot(mismatched_baseline, &settings, &original_regime);
        assert_eq!(
            mismatched.edge_cost(PrimitiveId::WALK_CARDINAL, features, 99),
            99
        );
        assert_eq!(mismatched.model_revision, 0);

        let mut changed = observation(&key, 50, 10, ObservationOutcome::Success);
        changed.context = changed_context;
        changed.id.journey_id = changed.context.journey_id;
        changed.id.generation = changed.context.generation;
        changed.id.leg = changed.context.leg;
        assert!(!profile.observe(&changed, &settings));
        assert_eq!(profile.data_revision, 3);
    }

    #[test]
    fn actor_and_control_contexts_fail_closed() {
        let key = ProfileKey::local_default();
        let mut invalid = context(&key);
        invalid.actor_capability_hash = key.capability_hash.wrapping_add(1);
        assert!(!invalid.validate());

        let first = follower_settings_hash(&FollowerSettings::default());
        let changed = FollowerSettings {
            turn_ease: 0.25,
            ..FollowerSettings::default()
        };
        assert_ne!(first, follower_settings_hash(&changed));
    }

    #[test]
    fn promotion_report_freezes_the_exact_bounded_candidate_and_settings() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = permissive_settings(AdaptiveMode::Enabled);
        for edge in 0..3 {
            assert!(profile.observe(
                &observation(&key, edge, 10, ObservationOutcome::Success),
                &settings,
            ));
        }
        let report = good_report(&profile, &settings, 1);
        let changed_settings = AdaptiveSettings {
            max_step_percent: 25,
            ..settings.clone()
        };
        assert_eq!(
            profile.promote(&report, &changed_settings),
            Err(PromotionError::StaleCandidate)
        );
        assert_eq!(profile.model_revision(), 0);
        assert_eq!(profile.promote(&report, &settings).unwrap(), 1);
    }

    #[test]
    fn safety_comparison_uses_rates_for_unequal_cohorts() {
        let settings = permissive_settings(AdaptiveMode::Shadow).validated();
        let report = PromotionReport {
            evaluation_id: 1,
            candidate_data_revision: 1,
            expected_model_revision: 0,
            candidate_hash: 1,
            known_good: JourneyMetrics {
                journeys: 100,
                failed: 0,
                p95_ticks: 100,
                damage_half_hearts: 10,
                setbacks: 0,
                corrections: 0,
                disconnects: 0,
            },
            shadow: JourneyMetrics {
                journeys: 20,
                failed: 0,
                p95_ticks: 100,
                damage_half_hearts: 3,
                setbacks: 0,
                corrections: 0,
                disconnects: 0,
            },
        };
        assert_eq!(
            report.validate(&settings),
            Err(PromotionError::SafetyRegression)
        );
    }

    #[test]
    fn shadow_candidate_preserves_fixed_safety_components() {
        let features = MoveFeatures::new(1, 0, TerrainClass::Water);
        let candidate = BTreeMap::from([(FeatureKey::exact(PrimitiveId::SWIM, features), 500_000)]);
        let snapshot = CostModelSnapshot {
            data_revision: 1,
            model_revision: 0,
            mode: AdaptiveMode::Shadow,
            baseline: MovementCosts::default(),
            active: BTreeMap::new(),
            candidate,
        };
        let baseline = CostComponents {
            time: 20,
            safety: 1_000,
            damage: 7,
            resource: 3,
            ..CostComponents::default()
        };
        let proposed = snapshot
            .proposed_edge_components(PrimitiveId::SWIM, features, baseline)
            .unwrap();
        assert_eq!(proposed.time, 10);
        assert_eq!(proposed.safety, 1_000);
        assert_eq!(proposed.damage, 7);
        assert_eq!(proposed.resource, 3);
        assert_eq!(proposed.total(), 1_020);

        let candidate_view = snapshot.candidate_view();
        assert!(candidate_view.affects_routing());
        assert_eq!(
            candidate_view
                .edge_components(PrimitiveId::SWIM, features, baseline)
                .total(),
            proposed.total()
        );
    }

    #[test]
    fn eviction_uses_profile_arrival_order_not_follower_local_sequences() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = AdaptiveSettings {
            max_buckets: 1,
            ..permissive_settings(AdaptiveMode::Shadow)
        };
        let full = observation(&key, 0, 5, ObservationOutcome::Success);
        assert!(profile.observe(&full, &settings));

        let mut partial = observation(&key, 1, 5, ObservationOutcome::Success);
        partial.features.terrain = TerrainClass::PartialBlock;
        partial.completion_sequence = full.completion_sequence;
        assert!(profile.observe(&partial, &settings));
        assert!(!profile.estimates.contains_key(&FeatureKey::exact(
            PrimitiveId::WALK_CARDINAL,
            full.features
        )));
        assert!(profile.estimates.contains_key(&FeatureKey::exact(
            PrimitiveId::WALK_CARDINAL,
            partial.features
        )));
    }

    #[test]
    fn promoted_model_remains_valid_after_its_estimate_bucket_is_evicted() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = AdaptiveSettings {
            max_buckets: 1,
            ..permissive_settings(AdaptiveMode::Enabled)
        };
        for edge in 0..3 {
            assert!(profile.observe(
                &observation(&key, edge, 10, ObservationOutcome::Success),
                &settings
            ));
        }
        let report = good_report(&profile, &settings, 1);
        profile.promote(&report, &settings).unwrap();

        let mut different = observation(&key, 20, 5, ObservationOutcome::Success);
        different.features.terrain = TerrainClass::PartialBlock;
        assert!(profile.observe(&different, &settings));
        assert!(profile.validate());

        let root = std::env::temp_dir().join(format!(
            "azalea-pathfinder-evicted-promoted-{}-{}",
            std::process::id(),
            next_observation_nonce()
        ));
        save_profile_atomic(&root, &profile).unwrap();
        assert_eq!(load_profile(&root, &key).unwrap().model_revision(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn persisted_models_cannot_bypass_multiplier_bucket_caps() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        assert!(profile.observe(
            &observation(&key, 0, 5, ObservationOutcome::Success),
            &AdaptiveSettings::default()
        ));
        profile.active.revision = 1;
        profile.active.source_data_revision = profile.data_revision;
        profile.last_evaluation_id = 1;
        for index in 0..=(MAX_PROFILE_BUCKETS + 11) {
            let features = MoveFeatures::new(
                (index % 17) as u8,
                (index / 17) as i8,
                TerrainClass::FullBlock,
            );
            profile.active.multipliers.insert(
                FeatureKey::exact(PrimitiveId::WALK_CARDINAL, features),
                MULTIPLIER_ONE,
            );
        }
        assert_eq!(profile.active.multipliers.len(), MAX_PROFILE_BUCKETS + 12);
        assert!(!profile.validate());
    }

    #[test]
    fn observations_older_than_the_replay_window_are_rejected() {
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let settings = permissive_settings(AdaptiveMode::Shadow);
        let mut newest = observation(&key, 0, 5, ObservationOutcome::Success);
        newest.id.nonce = (MAX_RECENT_COMPLETIONS as u64) + 100;
        assert!(profile.observe(&newest, &settings));

        let mut reordered = observation(&key, 1, 5, ObservationOutcome::Success);
        reordered.id.nonce = newest.id.nonce - 1;
        assert!(profile.observe(&reordered, &settings));

        let mut old = observation(&key, 2, 5, ObservationOutcome::Success);
        old.id.nonce = 100;
        assert!(!profile.observe(&old, &settings));
        assert_eq!(profile.data_revision, 2);
    }

    #[test]
    fn loading_a_future_nonce_floor_resumes_above_it() {
        let root = std::env::temp_dir().join(format!(
            "azalea-pathfinder-nonce-resume-{}-{}",
            std::process::id(),
            next_observation_nonce()
        ));
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        let mut future = observation(&key, 0, 5, ObservationOutcome::Success);
        future.id.nonce = next_observation_nonce().saturating_add(1_000_000);
        let floor = future.id.nonce;
        assert!(profile.observe(&future, &AdaptiveSettings::default()));
        save_profile_atomic(&root, &profile).unwrap();
        let loaded = load_profile(&root, &key).unwrap();
        assert_eq!(loaded.replay_high_water, floor);
        assert!(next_observation_nonce() > floor);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn profile_round_trip_replaces_existing_and_fails_closed() {
        let root = std::env::temp_dir().join(format!(
            "azalea-pathfinder-profile-test-{}-{}",
            std::process::id(),
            ProfileKey::local_default().stable_hash()
        ));
        let _ = fs::remove_dir_all(&root);
        let key = ProfileKey::local_default();
        let mut profile = AdaptiveProfile::new(key.clone());
        profile.observe(
            &observation(&key, 0, 5, ObservationOutcome::Success),
            &AdaptiveSettings::default(),
        );
        save_profile_atomic(&root, &profile).unwrap();
        profile.observe(
            &observation(&key, 1, 5, ObservationOutcome::Success),
            &AdaptiveSettings::default(),
        );
        save_profile_atomic(&root, &profile).unwrap();
        let loaded = load_profile(&root, &key).unwrap();
        assert_eq!(loaded.data_revision, 2);

        fs::write(profile_path(&root, &key), b"{truncated").unwrap();
        assert!(load_profile(&root, &key).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn oversized_profile_is_rejected_without_unbounded_read() {
        let root = std::env::temp_dir().join(format!(
            "azalea-pathfinder-oversized-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let key = ProfileKey::local_default();
        let file = File::create(profile_path(&root, &key)).unwrap();
        file.set_len(MAX_PROFILE_BYTES + 1).unwrap();
        assert!(matches!(
            load_profile(&root, &key),
            Err(ProfileError::TooLarge)
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn bounded_queue_counts_drops_and_worker_rejects_cross_profile() {
        let root = std::env::temp_dir().join(format!(
            "azalea-pathfinder-telemetry-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let key = ProfileKey::local_default();
        let worker = TelemetryWorker::spawn(root.clone(), key.clone(), 4).unwrap();
        assert!(
            TelemetryWorker::spawn(root.clone(), key.clone(), 4).is_err(),
            "only one writer may own a profile"
        );
        let queue = worker.queue();
        let mut wrong = observation(&key, 0, 5, ObservationOutcome::Success);
        wrong.context.profile.server = "another-server".into();
        assert!(queue.try_send(wrong));
        assert!(queue.try_send(observation(&key, 1, 5, ObservationOutcome::Success)));
        worker.shutdown_blocking();
        assert_eq!(queue.dropped(), 1);
        assert!(!queue.try_send(observation(&key, 2, 5, ObservationOutcome::Success)));
        assert_eq!(queue.dropped(), 2);
        let path = fs::canonicalize(&root).unwrap().join(format!(
            "telemetry-{}-{}.jsonl",
            key.digest().to_hex(),
            std::process::id()
        ));
        let lines = fs::read_to_string(path).unwrap();
        assert_eq!(lines.lines().count(), 1);
        let _ = fs::remove_dir_all(root);
    }
}
