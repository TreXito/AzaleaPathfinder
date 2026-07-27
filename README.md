# azalea-pathfinder

Path planning and movement for [Azalea](https://github.com/azalea-rs/azalea) clients.

It provides:

- local A* pathfinding over a snapshot of the loaded world
- walking, auto-stepping, jumping, gap parkour, ladder climbing, swimming, and falling
- lava and water policies with configurable movement costs
- partial paths when the destination is not loaded yet
- resumable/anytime A* with world, goal, and cost revision checks
- a tick-driven path follower with pause, cancel, and replan support
- opt-in, shadow-first movement telemetry and bounded cost calibration
- scoped dynamic-obstacle overlays and conservative path invalidation
- a small high-level router for travelling between areas

## Install

```toml
[dependencies]
azalea = { git = "https://github.com/RaymondShell/azalea", branch = "main" }
azalea-pathfinder = { git = "https://github.com/RaymondShell/AzaleaPathfinder", branch = "master" }
bevy_ecs = "0.19"
```

`bevy_ecs` must match the version used by Azalea. Azalea's NBT dependency uses
nightly-only portable SIMD, so this repository pins a known-good nightly in
`rust-toolchain.toml`. Use a release build for live navigation; large searches
are noticeably slower in debug builds. Run `cargo update -p azalea` before
building to refresh the tracked Azalea branch.

## Quick start

Register the plugin with your client:

```rust
use azalea::prelude::*;
use azalea_pathfinder::{
    AzaleaPathfinderClientExt, AzaleaPathfinderPlugin, NavigationGoal,
    NavigationRequest,
};

ClientBuilder::new()
    .add_plugins(AzaleaPathfinderPlugin)
    .set_handler(handle)
    .start(account, "server.address")
    .await?;
```

Start navigation from a client handler:

```rust
let generation = 1;

client.start_navigation(NavigationRequest {
    goal: NavigationGoal::Fixed(BlockPos::new(120, 70, -340)),
    path_seed: 0,
    generation,
});

let status = client.wait_for_navigation(generation).await;
```

`generation` identifies the request, so results from an older plan cannot replace
a newer one. Use `pause_navigation(true)`, `pause_navigation(false)`, or
`cancel_navigation()` to control the active route.

## Planning a path directly

Capture an area containing both endpoints, then plan against the snapshot. The
world lock is only held while the snapshot is copied.

```rust
use azalea::BlockPos;
use azalea_pathfinder::{
    MoveContext, WorldSnapshot, default_moves, find_path_best_effort,
};

let world = client.world()?;
let start = BlockPos::from(client.position()?);
let goal = BlockPos::new(120, 70, -340);
let margin: i32 = 16;

let lo = BlockPos::new(
    start.x.min(goal.x).saturating_sub(margin),
    start.y.min(goal.y).saturating_sub(margin),
    start.z.min(goal.z).saturating_sub(margin),
);
let hi = BlockPos::new(
    start.x.max(goal.x).saturating_add(margin),
    start.y.max(goal.y).saturating_add(margin),
    start.z.max(goal.z).saturating_add(margin),
);

let snapshot = WorldSnapshot::try_capture(&world, lo, hi)?;
let moves = default_moves();
let context = MoveContext::default();
let (path, reached_goal) =
    find_path_best_effort(&snapshot, start, goal, &moves, &context);

for node in &path.nodes {
    println!("{:?} via {:?}", node.pos, node.reached_by);
}
```

When `reached_goal` is false, the path ends at the closest reachable point. Follow
it to load more chunks, take another snapshot, and plan again. Use `find_path` if
you want a strict `Result<Path, PathError>` instead.

### Resumable search and custom goals

`SearchSession` retains its frontier and caches across small work slices. A
zero-expansion slice is side-effect free; cumulative expansion and compute-time
limits still come from `MoveContext`.

```rust
use azalea_pathfinder::{
    BlockGoal, SearchSession, SearchSlice, SearchStatus,
};

let goal = BlockGoal::new(goal, context.goal_tolerance);
let mut search = SearchSession::new(&snapshot, start, &goal, &moves, &context);

loop {
    match search.advance(SearchSlice::new(512)) {
        SearchStatus::InProgress => continue,
        SearchStatus::Found(path) => break Some(path),
        SearchStatus::Exhausted | SearchStatus::BudgetExhausted => {
            break Some(search.best_path());
        }
        SearchStatus::Invalidated(_) => break None,
    }
}
```

Implement the object-safe `Goal` trait for area, entity, or composite goals.
Return `None` from `heuristic_lower_bound` unless it is a proven lower bound;
the planner then safely uses Dijkstra ordering. Use `advance_checked` with
`SearchAssumptions` when an external world snapshot or promoted cost model has
a revision.

## Adaptive costs

Adaptation is disabled for routing unless all of these happen:

1. the plugin is given an explicit `ProfileKey` describing server, protocol,
   dimension, capabilities, latency bucket, and manually supplied environment;
2. observations meet minimum sample and confidence gates;
3. a paired `PromotionReport` shows no failure, damage, setback, correction,
   disconnect, or allowed p95 regression;
4. the candidate is explicitly promoted; and
5. `AdaptiveMode::Enabled` is selected.

The default is shadow mode with no profile. Shadow data can propose costs but
cannot alter a route. Each promotion is step-clamped, retains the prior
known-good model, and can be rolled back with a `GuardSignal`. Learning uses
dimensionless relative multipliers, so observed ticks are never mixed with the
planner's tenths-of-a-block cost unit. Hard constraints such as lava policy,
collision checks, and survivable fall distance are outside the model.

The library does not manufacture or certify the paired journey metrics in a
`PromotionReport`. An operator or external evaluator must run the known-good
and exact `candidate_view()` models on comparable journeys, supply those
metrics, and request promotion. This is guarded online calibration, not an
autonomous deployment system.

`MotionPlan` carries stable primitive IDs and versioned features alongside the
legacy `Path`, without changing `Path` or `PathNode`. `ObservedPathFollower`
owns a begin-to-terminal attempt lifecycle: pauses do not consume active time;
cursor skips, re-anchors, corrections and cancellations are interruptions, not
successes; duplicate completions are rejected. Observed execution pins
line-of-sight smoothing to the immediate node so one timed primitive cannot
physically cut across several path edges. If hunger prevents requested
sprinting, telemetry is disabled for the rest of that leg.

Profiles are bounded, versioned, context-checked and atomically replaced on
Windows and Unix. A model is used only when build ID, configured planner
policy, effective follower controls, baseline costs and actor capability all
match its observation regime. Use `AdaptivePathfinderRuntime::load_from` or
`install_profile` with an explicit replacement policy to restore a persisted
model.

`TelemetryWorker` is optional and bounded. It owns one writer per profile per
process, uses process-specific JSONL streams to avoid cross-process
append/rotation races, and performs file I/O off the game tick. Use
`shutdown_blocking` only from maintenance or shutdown code.

## Routing between areas

`WorldGraph` handles coarse travel such as server warps. Local A* handles the walk
within a loaded area.

```rust
use azalea_pathfinder::{WorldGraph, route};

let graph = WorldGraph::skyblock_default();
let from = graph.place_by_mode("hub").unwrap();
let to = graph.place_index("dwarven_mines").unwrap();

if let Some(steps) = route(&graph, from, to) {
    for step in steps {
        println!("{:?}", step.edge);
    }
}
```

An empty route means the client is already at the destination. `None` means the
two places are not connected.

## Configuration

`MoveContext::default()` is deliberately cautious. Its main settings are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `max_fall` | 10 | Largest candidate drop; damage is priced separately |
| `fall_damage_penalty` | 45 | Extra cost per half-heart of fall damage |
| `goal_tolerance` | 1 | Manhattan distance accepted as arrival |
| `max_expansions` | 150,000 | Search node limit |
| `time_budget_ms` | 2,000 | Search time limit |
| `wall_penalty` | 2 | Preference for open space |
| `lava_policy` | forbidden, 2-block clearance | Lava safety rule |
| `water_policy` | forbidden | Water safety rule |
| `grazing_step_penalty` | 50 | Avoid uneven fractional-height edges |
| `path_seed` | 0 | Tie-breaking between equal-cost routes |

The default movement costs are expressed in tenths of a block: cardinal walking
costs 10, diagonal walking 14, auto-stepping 12, jumping 24, parkour 22 per
horizontal block, climbing 30, swimming 20, and falling starts at 14 plus 6 per
block.

Snapshots are rejected when their inclusive volume exceeds 2,000,000 blocks.
Planning time includes both snapshot capture and A* search. Lava scan radii are
clamped to 32 blocks to keep caller-provided settings from causing unbounded
per-node work.

The plugin also exposes `PathfinderSettings` for snapshot size, follower behaviour,
partial-path limits, and optional periodic replanning.

For plugin-driven navigation, `max_fall` is additionally capped from the
entity's current `Health` with a two-heart reserve. Armor and potion effects are
not credited because they are not yet represented in the fall model.

Set `PF_NAV_DEBUG` to trace navigation legs or `PF_PARKOUR_DEBUG` to trace gap
candidate generation. If `PF_PATH_DIR` is set, each accepted plan is exported
to a sanitized `bot-<name>.path` file in that directory for visualization.

## Design notes

The planner reads blocks through the `WorldView` trait. `WorldSnapshot` is the
normal choice for live clients because it lets the search run without holding the
world lock. Unloaded blocks are impassable, while lava is forbidden by default.

Snapshot capture intentionally batches the Azalea world lock. It is therefore
an immutable, bounded, approximately contemporaneous view rather than an atomic
world instant. Revision checks detect application-known invalidation, and the
follower still validates every live segment before executing it.

Movement rules implement `Move`. Built-in one-block moves use the A* heuristic.
Custom moves fall back to Dijkstra unless they explicitly declare that they obey
the heuristic's assumptions, which keeps long-range or unusually cheap moves
optimal.

`DynamicObstacleOverlay` scopes transient hard and soft obstacles to one
server/world/dimension and expires them by tick. `PathDependencyIndex` tracks
support, body/head clearance, diagonal corners, jump arcs, landings, climbable
blocks and nearby hazards, returning the earliest affected edge after a block
change. Path joining is structural only unless the caller supplies current
scope/revision and live edge validation; this project does not claim D* Lite.
These resilience and resumable-search types are low-level opt-in building
blocks. The bundled high-level plugin still reacts to live validation failures
with a fresh bounded plan; it does not yet wire entity feeds into the overlay or
perform incremental path splicing automatically.

## Current limitations

- AOTV and Etherwarp are extension points only; they do not generate moves yet.
- The bundled SkyBlock graph is a small warp-based starting map, not a complete
  map of every island and transition.
- Planning only knows about loaded blocks included in the snapshot.
- Dynamic obstacle feeds, dependency-driven repair, and validated splicing are
  library APIs rather than automatic high-level plugin behavior.
- Water movement exists, but it is forbidden by default while the current
  Azalea water physics disagrees with the target server simulation.

## Testing

```sh
cargo update -p azalea
cargo test
cargo clippy --all-targets -- -D warnings
cargo bench --bench pathfinding
```

See [SECURITY.md](SECURITY.md) for the currently accepted upstream RSA advisory
and the dependency-review policy.

## License

Licensed under the [MIT License](LICENSE).
