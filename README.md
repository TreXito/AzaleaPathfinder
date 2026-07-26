# azalea-pathfinder

Path planning and movement for [Azalea](https://github.com/azalea-rs/azalea) clients.

It provides:

- local A* pathfinding over a snapshot of the loaded world
- walking, auto-stepping, jumping, gap parkour, ladder climbing, swimming, and falling
- lava and water policies with configurable movement costs
- partial paths when the destination is not loaded yet
- a tick-driven path follower with pause, cancel, and replan support
- a small high-level router for travelling between areas

## Install

```toml
[dependencies]
azalea = { git = "https://github.com/RaymondShell/azalea", rev = "c3094e4b92c8856a611d618da692b8decf315d87" }
azalea-pathfinder = { git = "https://github.com/RaymondShell/AzaleaPathfinder", rev = "<release-commit>" }
bevy_ecs = "0.19"
```

`bevy_ecs` must match the version used by Azalea. Azalea's NBT dependency uses
nightly-only portable SIMD, so this repository pins a known-good nightly in
`rust-toolchain.toml`. Use a release build for live navigation; large searches
are noticeably slower in debug builds.

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

Set `PF_NAV_DEBUG` to trace navigation legs or `PF_PARKOUR_DEBUG` to trace gap
candidate generation. If `PF_PATH_DIR` is set, each accepted plan is exported
to a sanitized `bot-<name>.path` file in that directory for visualization.

## Design notes

The planner reads blocks through the `WorldView` trait. `WorldSnapshot` is the
normal choice for live clients because it lets the search run without holding the
world lock. Unloaded blocks are impassable, while lava is forbidden by default.

Movement rules implement `Move`. Built-in one-block moves use the A* heuristic.
Custom moves fall back to Dijkstra unless they explicitly declare that they obey
the heuristic's assumptions, which keeps long-range or unusually cheap moves
optimal.

## Current limitations

- AOTV and Etherwarp are extension points only; they do not generate moves yet.
- The bundled SkyBlock graph is a small warp-based starting map, not a complete
  map of every island and transition.
- Planning only knows about loaded blocks included in the snapshot.
- Water movement exists, but it is forbidden by default while the pinned
  Azalea water physics disagrees with the target server simulation.

## Testing

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

See [SECURITY.md](SECURITY.md) for the currently accepted upstream RSA advisory
and the dependency-review policy.

## License

Licensed under the [MIT License](LICENSE).
