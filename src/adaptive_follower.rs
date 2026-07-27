//! Exactly-once observation lifecycle around the compatibility path follower.

use crate::adaptive::{
    AttributionEvidence, InterruptionReason, MoveAttempt, MoveObservation, ObservationContext,
    ObservationId, ObservationOutcome, next_observation_nonce,
};
use crate::planning::{MotionPlan, MotionStep};
use crate::{
    FollowerDirective, FollowerFailure, FollowerFrame, FollowerProgress, FollowerSettings, Path,
    PathFollower, WorldView,
};

struct ActiveAttempt {
    edge_index: usize,
    attempt: MoveAttempt,
}

/// A [`PathFollower`] that emits structured observations without changing its
/// steering API. Pauses retain the active attempt and do not consume its clock;
/// cancellation, correction, re-anchor, unsafe validation and stalls each
/// produce one explicit terminal observation.
pub struct ObservedPathFollower {
    follower: PathFollower,
    steps: Vec<MotionStep>,
    context: ObservationContext,
    active: Option<ActiveAttempt>,
    attempts: Vec<u16>,
    observations: Vec<MoveObservation>,
    active_ticks: u64,
    completion_sequence: u64,
    terminal: Option<FollowerDirective>,
    observations_enabled: bool,
}

impl ObservedPathFollower {
    pub fn new(
        plan: MotionPlan,
        settings: FollowerSettings,
        seed: u64,
        context: ObservationContext,
    ) -> Self {
        let (path, steps) = plan.into_parts();
        let attempts = vec![0; steps.len()];
        let settings = observed_follower_settings(settings);
        Self {
            follower: PathFollower::new(path, settings, seed),
            steps,
            context,
            active: None,
            attempts,
            observations: Vec::new(),
            active_ticks: 0,
            completion_sequence: 0,
            terminal: None,
            observations_enabled: true,
        }
    }

    pub fn path(&self) -> &Path {
        self.follower.path()
    }

    pub fn current_node_index(&self) -> usize {
        self.follower.current_node_index()
    }

    pub fn model_revision(&self) -> u64 {
        self.context.model_revision
    }

    pub fn tick(&mut self, world: &dyn WorldView, frame: FollowerFrame) -> FollowerDirective {
        if let Some(terminal) = self.terminal {
            return terminal;
        }
        if !frame.paused {
            self.begin_current_attempt();
            self.active_ticks = self.active_ticks.saturating_add(1);
        }
        let directive = self.follower.tick(world, frame);
        let progress = self.follower.take_progress();
        let failure = self.follower.take_failure();

        match progress {
            FollowerProgress::None => {}
            FollowerProgress::Advanced { from, to } => {
                let expected_edge = from.saturating_sub(1);
                if to == from.saturating_add(1)
                    && self
                        .active
                        .as_ref()
                        .is_some_and(|active| active.edge_index == expected_edge)
                {
                    self.finish_active(
                        ObservationOutcome::Success,
                        AttributionEvidence::ReachedPlannedNode,
                    );
                } else {
                    // Advancing over more than one cursor node is a smoothing
                    // or correction event. None of the skipped primitives is
                    // credited as a success.
                    self.finish_active(
                        ObservationOutcome::Interrupted(InterruptionReason::Unknown),
                        AttributionEvidence::ExternalMovementEvent,
                    );
                }
            }
            FollowerProgress::Reanchored { .. } => {
                self.finish_active(
                    ObservationOutcome::Interrupted(InterruptionReason::ExternalImpulse),
                    AttributionEvidence::ExternalMovementEvent,
                );
            }
        }

        match directive {
            FollowerDirective::Arrived => {
                // The final target does not advance the follower cursor: it is
                // already the current node. Arrival is its terminal evidence.
                self.finish_active(
                    ObservationOutcome::Success,
                    AttributionEvidence::ReachedPlannedNode,
                );
                self.terminal = Some(directive);
            }
            FollowerDirective::Stuck { .. } => {
                let (outcome, evidence) = match failure {
                    Some(FollowerFailure::TimedOut) => (
                        ObservationOutcome::TimedOut,
                        AttributionEvidence::CumulativeDeadline,
                    ),
                    Some(FollowerFailure::FellOff) => (
                        ObservationOutcome::FellOff,
                        AttributionEvidence::FellBelowPath,
                    ),
                    Some(FollowerFailure::Stalled) => (
                        ObservationOutcome::Stalled,
                        AttributionEvidence::StallDetector,
                    ),
                    None => (
                        ObservationOutcome::Interrupted(InterruptionReason::Unknown),
                        AttributionEvidence::ExternalMovementEvent,
                    ),
                };
                self.finish_active(outcome, evidence);
                self.terminal = Some(directive);
            }
            FollowerDirective::Unsafe { .. } => {
                self.finish_active(
                    ObservationOutcome::Interrupted(InterruptionReason::WorldChanged),
                    AttributionEvidence::ExternalMovementEvent,
                );
                self.terminal = Some(directive);
            }
            FollowerDirective::Paused
            | FollowerDirective::Move { .. }
            | FollowerDirective::Wait { .. } => {}
        }
        directive
    }

    /// Explicitly terminates an active attempt when its owning navigation is
    /// replaced, cancelled, invalidated or externally corrected.
    pub fn interrupt(&mut self, reason: InterruptionReason) {
        if self.terminal.is_some() {
            return;
        }
        let evidence = interruption_evidence(reason);
        self.finish_active(ObservationOutcome::Interrupted(reason), evidence);
        // An interrupt transfers ownership away from this executor. Latch a
        // non-moving terminal directive so an accidentally retained wrapper
        // cannot start a fresh attempt if ticked again.
        self.terminal = Some(FollowerDirective::Paused);
    }

    /// Permanently suppresses telemetry for the rest of this leg after an
    /// unmodelled execution condition (for example, hunger preventing an
    /// requested sprint). Steering continues normally, but a partial edge is
    /// never resumed and misattributed after the condition clears.
    pub fn disable_observations(&mut self, reason: InterruptionReason) {
        if !self.observations_enabled {
            return;
        }
        self.finish_active(
            ObservationOutcome::Interrupted(reason),
            interruption_evidence(reason),
        );
        self.observations_enabled = false;
    }

    pub fn drain_observations(&mut self) -> impl Iterator<Item = MoveObservation> + '_ {
        self.observations.drain(..)
    }

    fn begin_current_attempt(&mut self) {
        if !self.observations_enabled || self.active.is_some() || self.steps.is_empty() {
            return;
        }
        let node_index = self.follower.current_node_index();
        let Some(edge_index) = node_index.checked_sub(1) else {
            return;
        };
        let Some(step) = self.steps.get(edge_index) else {
            return;
        };
        let attempt_number = self.attempts[edge_index];
        let Some(next_attempt_number) = attempt_number.checked_add(1) else {
            self.terminal = Some(FollowerDirective::Paused);
            return;
        };
        let id = ObservationId {
            nonce: next_observation_nonce(),
            journey_id: self.context.journey_id,
            generation: self.context.generation,
            leg: self.context.leg,
            edge_index: step.edge_index,
            attempt: attempt_number,
        };
        let Ok(attempt) = MoveAttempt::begin(
            self.context.clone(),
            id,
            step.primitive,
            step.features,
            step.from,
            step.to,
            step.predicted,
            step.predicted_ticks,
            self.active_ticks,
        ) else {
            return;
        };
        self.attempts[edge_index] = next_attempt_number;
        self.active = Some(ActiveAttempt {
            edge_index,
            attempt,
        });
    }

    fn finish_active(&mut self, outcome: ObservationOutcome, evidence: AttributionEvidence) {
        let Some(active) = self.active.take() else {
            return;
        };
        let Some(completion_sequence) = self.completion_sequence.checked_add(1) else {
            self.terminal = Some(FollowerDirective::Paused);
            return;
        };
        self.completion_sequence = completion_sequence;
        if let Ok(observation) =
            active
                .attempt
                .finish(self.active_ticks, completion_sequence, outcome, evidence)
        {
            self.observations.push(observation);
        }
    }
}

fn interruption_evidence(reason: InterruptionReason) -> AttributionEvidence {
    match reason {
        InterruptionReason::Pause
        | InterruptionReason::Cancelled
        | InterruptionReason::GoalChanged => AttributionEvidence::PauseOrCancellation,
        InterruptionReason::WorldChanged
        | InterruptionReason::ExternalImpulse
        | InterruptionReason::ServerCorrection
        | InterruptionReason::Unknown => AttributionEvidence::ExternalMovementEvent,
    }
}

/// Exact primitive attribution requires the follower to steer at the immediate
/// planned node. Normal LOS smoothing can physically execute a multi-edge
/// corner cut while the telemetry wrapper is timing only the first edge.
pub fn observed_follower_settings(mut settings: FollowerSettings) -> FollowerSettings {
    settings.max_los_skip = 0;
    settings
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use azalea::{BlockPos, Vec3};

    use super::*;
    use crate::adaptive::{ProfileKey, movement_costs_hash};
    use crate::local::moves::{MoveContext, MovementCosts};
    use crate::local::world::BlockKind;
    use crate::{MoveKind, PathNode};

    struct Grid(HashMap<(i32, i32, i32), BlockKind>);

    impl WorldView for Grid {
        fn block(&self, position: BlockPos) -> BlockKind {
            self.0
                .get(&(position.x, position.y, position.z))
                .copied()
                .unwrap_or(BlockKind::Air)
        }
    }

    fn context() -> ObservationContext {
        let profile = ProfileKey::local_default();
        ObservationContext {
            actor_capability_hash: profile.capability_hash,
            profile,
            journey_id: 1,
            generation: 2,
            leg: 1,
            plan_revision: 1,
            world_revision: 1,
            model_revision: 0,
            planner_settings_hash: 3,
            control_settings_hash: 4,
            baseline_costs_hash: movement_costs_hash(MovementCosts::default()),
            build_id: "test".into(),
            created_unix_ms: 0,
        }
    }

    fn plan(world: &dyn WorldView) -> MotionPlan {
        MotionPlan::try_from_path(
            Path {
                nodes: vec![
                    PathNode {
                        pos: BlockPos::new(0, 64, 0),
                        reached_by: MoveKind::Start,
                    },
                    PathNode {
                        pos: BlockPos::new(1, 64, 0),
                        reached_by: MoveKind::Walk,
                    },
                ],
                total_cost: 10,
            },
            world,
            &MoveContext::default(),
        )
        .unwrap()
    }

    fn frame(x: f64, paused: bool) -> FollowerFrame {
        FollowerFrame {
            position: Vec3 { x, y: 64.0, z: 0.5 },
            on_ground: true,
            horizontal_collision: false,
            paused,
        }
    }

    fn settings() -> FollowerSettings {
        FollowerSettings {
            arrival_radius_xz: 0.2,
            close_enough_xz: 0.2,
            ..FollowerSettings::default()
        }
    }

    #[test]
    fn final_arrival_emits_one_success() {
        let world = Grid(HashMap::new());
        let mut follower = ObservedPathFollower::new(plan(&world), settings(), 0, context());
        assert_eq!(
            follower.tick(&world, frame(1.5, false)),
            FollowerDirective::Arrived
        );
        assert_eq!(
            follower.tick(&world, frame(1.5, false)),
            FollowerDirective::Arrived
        );
        let observations: Vec<_> = follower.drain_observations().collect();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].outcome, ObservationOutcome::Success);
        assert_eq!(observations[0].actual_ticks(), 1);
    }

    #[test]
    fn pause_does_not_finish_or_consume_attempt_time() {
        let world = Grid(HashMap::new());
        let mut follower = ObservedPathFollower::new(plan(&world), settings(), 0, context());
        assert!(matches!(
            follower.tick(&world, frame(0.5, false)),
            FollowerDirective::Move { .. }
        ));
        assert_eq!(
            follower.tick(&world, frame(0.5, true)),
            FollowerDirective::Paused
        );
        assert_eq!(follower.drain_observations().count(), 0);
        follower.tick(&world, frame(1.5, false));
        let observation = follower.drain_observations().next().unwrap();
        assert_eq!(observation.actual_ticks(), 2);
    }

    #[test]
    fn cancellation_finishes_once_without_failure_attribution() {
        let world = Grid(HashMap::new());
        let mut follower = ObservedPathFollower::new(plan(&world), settings(), 0, context());
        follower.tick(&world, frame(0.5, false));
        follower.interrupt(InterruptionReason::Cancelled);
        follower.interrupt(InterruptionReason::Cancelled);
        assert_eq!(
            follower.tick(&world, frame(0.5, false)),
            FollowerDirective::Paused
        );
        assert_eq!(
            follower.tick(&world, frame(1.5, false)),
            FollowerDirective::Paused
        );
        let observations: Vec<_> = follower.drain_observations().collect();
        assert_eq!(observations.len(), 1);
        assert_eq!(
            observations[0].outcome,
            ObservationOutcome::Interrupted(InterruptionReason::Cancelled)
        );
    }

    #[test]
    fn observed_execution_pins_a_corner_to_the_immediate_node() {
        let world = Grid(
            [
                ((0, 63, 0), BlockKind::Solid),
                ((1, 63, 0), BlockKind::Solid),
                ((1, 63, 1), BlockKind::Solid),
            ]
            .into_iter()
            .collect(),
        );
        let path = Path {
            nodes: vec![
                PathNode {
                    pos: BlockPos::new(0, 64, 0),
                    reached_by: MoveKind::Start,
                },
                PathNode {
                    pos: BlockPos::new(1, 64, 0),
                    reached_by: MoveKind::Walk,
                },
                PathNode {
                    pos: BlockPos::new(1, 64, 1),
                    reached_by: MoveKind::Walk,
                },
            ],
            total_cost: 20,
        };
        let plan = MotionPlan::try_from_path(path, &world, &MoveContext::default()).unwrap();
        let mut follower = ObservedPathFollower::new(plan, settings(), 0, context());
        let FollowerDirective::Move { target, .. } = follower.tick(&world, frame(0.5, false))
        else {
            panic!("expected observed follower to move");
        };
        assert_eq!(target, Vec3::new(1.5, 64.0, 0.5));
    }

    #[test]
    fn unmodelled_execution_disables_the_remainder_of_the_leg() {
        let world = Grid(HashMap::new());
        let mut follower = ObservedPathFollower::new(plan(&world), settings(), 0, context());
        assert!(matches!(
            follower.tick(&world, frame(0.5, false)),
            FollowerDirective::Move { .. }
        ));
        follower.disable_observations(InterruptionReason::Unknown);
        follower.disable_observations(InterruptionReason::Unknown);
        assert_eq!(
            follower.tick(&world, frame(1.5, false)),
            FollowerDirective::Arrived
        );
        let observations: Vec<_> = follower.drain_observations().collect();
        assert_eq!(observations.len(), 1);
        assert_eq!(
            observations[0].outcome,
            ObservationOutcome::Interrupted(InterruptionReason::Unknown)
        );
    }
}
