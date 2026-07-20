# azalea-pathfinder

Block-level pathfinding and high-level travel routing for [Azalea](https://github.com/azalea-rs/azalea) clients.

The bot decides **where** to go; this crate decides **how** to get there. It plans
routes with an admissible A\* search over a lock-free world snapshot, prices moves
for conservative (SkyBlock-calibrated) play, and treats lava as a hard hazard by
default. The planner is independent of any combat, farm, or application state, so
any Azalea client can plan against it without exposing its own internals.

## Status

| Piece | State |
| --- | --- |
| Local A\* planner (`find_path`, `find_path_best_effort`) | ✅ Ready, regression tested |
| World snapshot capture (`WorldSnapshot`) | ✅ Ready |
| High-level place graph + Dijkstra router (`WorldGraph`, `route`) | ✅ Ready (warp-only starter graph) |
| Movement cost model + lava safety (`MoveContext`, `LavaPolicy`) | ✅ Ready |
| Shared tick-driven path follower (`PathFollower`) | ✅ Ready |
| `AzaleaPathfinderPlugin` execution loop | ✅ Ready — plan, follow, pause, cancel, replan, status |
| Item teleports (AOTV / Etherwarp) | Optional extension points, disabled by default |

`AzaleaPathfinderPlugin` is operational: a `NavigationRequest` plans on Azalea's
compute pool, follows the resulting path on game ticks, incrementally replans partial
routes, honors `NavigationPaused`, and publishes `NavigationStatus`. Applications
with additional policy can also use the same `PathFollower` engine directly.

## Features

- **Admissible A\*** with vertical guidance, so the search is pulled toward the
  goal's altitude instead of treating every Y-level as equal — and stays optimal.
- **Lock-free planning.** A brief read lock copies a bounded cube of the world into
  an owned dense array; the search then runs holding **no** lock, so a long search
  can never starve Azalea's tick.
- **Human-like movement.** Walks, cuts corners on diagonals, auto-steps slabs and
  stairs instead of hopping them, keeps clearance from walls, and drops off ledges.
- **Hard lava safety.** By default lava and a two-block clearance buffer are
  forbidden. If the bot starts inside the buffer, only exposure-reducing escape
  steps are allowed. A `Penalized` policy is available as an explicit opt-in.
- **Best-effort routing.** When the goal's chunks aren't loaded yet, the planner
  returns the closest reachable partial path so the caller can walk toward the goal
  (loading chunks) and re-plan.
- **Two-tier navigation.** A tiny place graph routes *between* areas (warps, pads),
  while the local A\* handles movement *within* the loaded world.
- **Extensible without touching the core.** New movement abilities are new `Move`
  implementations; new destinations are new graph data. The A\* core never changes.

## Installation

```toml
[dependencies]
azalea = { git = "https://github.com/RaymondShell/azalea", branch = "main" }
azalea-pathfinder = { git = "https://github.com/RaymondShell/AzaleaPathfinder", branch = "main" }
bevy_ecs = "0.19"
```

The direct `bevy_ecs` dependency is required (not optional): the `#[derive(Resource)]`
macro on `PathfinderSettings` resolves its crate path through it, and its version
must stay in lockstep with the `bevy_ecs` that Azalea uses (currently `0.19`).

Requires Rust 1.87+ (edition 2024). **Build in release mode for live use** — the
default search budget (150k node expansions) is generous and is much slower to
exhaust in a debug build.

## Quick start

### Register the plugin

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

From a client handler, start a route and optionally await its terminal status:

```rust
let generation = 1;
client.start_navigation(NavigationRequest {
    goal: NavigationGoal::Fixed(BlockPos::new(120, 70, -340)),
    path_seed: 0,
    generation,
});

let status = client.wait_for_navigation(generation).await;
```

`pause_navigation(true)` stops movement without consuming the stall budget;
`pause_navigation(false)` resumes, and `cancel_navigation()` stops and removes the
active request.

### Plan a local path

`WorldSnapshot::capture` copies a bounded region under one brief lock, then
`find_path_best_effort` searches over that owned snapshot.

```rust
use azalea::BlockPos;
use azalea_pathfinder::{default_moves, find_path_best_effort, MoveContext, WorldSnapshot};

// inside a handler holding a `Client`:
let world = client.world()?;                       // Arc<RwLock<World>>
let start = BlockPos::from(client.position()?);    // the bot's current block
let goal = BlockPos::new(120, 70, -340);

// Size the captured box to CONTAIN both start and goal (+ margin). A box that
// clips the goal or a corridor makes the search detour around its own edge.
let margin = 16;
let lo = BlockPos::new(
    start.x.min(goal.x) - margin,
    start.y.min(goal.y) - margin,
    start.z.min(goal.z) - margin,
);
let hi = BlockPos::new(
    start.x.max(goal.x) + margin,
    start.y.max(goal.y) + margin,
    start.z.max(goal.z) + margin,
);
let snapshot = WorldSnapshot::capture(&world, lo, hi);

let ctx = MoveContext::default();  // conservative, lava-forbidding SkyBlock defaults
let moves = default_moves();
let (path, reached) = find_path_best_effort(&snapshot, start, goal, &moves, &ctx);

for node in &path.nodes {
    println!("{:?} via {:?}", node.pos, node.reached_by);
}

if !reached {
    // Goal was outside the loaded/captured world. Walk this partial path to load
    // new chunks, then capture + plan again from the new position.
}
```

For a strict "succeed or fail" search (used mainly as the movement-rule test bed),
use `find_path`, which returns `Result<Path, PathError>` instead of a best-effort
partial.

### Route between areas

The place graph plans the high-level legs (which warps to take) before the local
planner handles on-foot movement within each area.

```rust
use azalea_pathfinder::{route, WorldGraph};

let graph = WorldGraph::skyblock_default();
let from = graph.place_by_mode("hub").unwrap();       // where the bot is (locraw mode)
let to = graph.place_index("dwarven_mines").unwrap(); // destination

// Cheapest sequence of legs; empty if already there, None if disconnected.
if let Some(steps) = route(&graph, from, to) {
    for step in steps {
        // step.edge      -> e.g. TravelEdge::Warp { name } → run `/warp <name>`
        // step.to_mode   -> the locraw mode expected after this leg (verify it landed)
        // step.to_anchor -> walk target for Walk edges (the destination's anchor)
    }
}
```

## How it works

Navigation is split into two layers with a clean boundary:

```
NavigationGoal ─▶ router::route  ──▶ high-level legs (warps, pads) between areas
                                     │
                                     ▼  (per leg, within the loaded world)
              WorldSnapshot::capture ─▶ find_path_best_effort ─▶ block-level Path
```

### World abstraction

The planner sees the world only through the `WorldView` trait
(`block(pos) -> BlockKind` plus a default `standable` rule). Blocks are classified
coarsely:

| `BlockKind` | Meaning |
| --- | --- |
| `Air` | Passable / occupiable |
| `Solid` | Full block: floor, wall, ceiling |
| `Step` | Bottom slab, bottom stairs, carpet — walked onto without jumping (auto-step) |
| `Fence` | Fence / wall / fence gate — 1.5 blocks tall: neither passable nor climbable, so the planner routes *around* it |
| `Lava` | Structurally occupiable so a trapped bot can plan an exit; the lava policy governs entry |
| `Unloaded` | Missing chunk data — impassable, so paths never leave the known world |

Two implementors ship: `WorldView for azalea::world::World` (queries the live world)
and `WorldSnapshot` (an owned, lock-free copy). **Prefer the snapshot for live use** —
searching over the live world holds the world read lock in a tight loop for the whole
search budget, which can starve the tick that needs `world.write()` each frame.

### Movement rules

Each way of moving is a `Move` implementation that pushes candidate edges. The
default set is walk, jump, and fall, plus two inert teleport extension points:

```rust
pub fn default_moves() -> Vec<Box<dyn Move>>; // Walk, Jump, Fall, (Aotv), (Etherwarp)
```

### Cost model

Costs are in tenths of a block. The defaults deliberately make risky or fidgety
movement more expensive than plain walking, so paths don't jitter to save a block:

| Move | Cost |
| --- | --- |
| Walk (cardinal) | 10 |
| Walk (diagonal) | 14 |
| Auto-step (slab/stair) | 12 |
| Jump up one block | 24 |
| Fall | 14 + 6 per block dropped |
| Warp (router) | 200 |

On top of edge costs, a node accrues terrain penalties: a per-adjacent-wall toll
(drift toward open space) and a distance-weighted lava-proximity toll beyond the
hard buffer. These only ever *add* cost, so the A\* heuristic stays admissible.

### Tuning (`MoveContext`)

`MoveContext::default()` is calibrated for cautious SkyBlock play. Notable knobs:

| Field | Default | Purpose |
| --- | --- | --- |
| `max_fall` | 3 | Max blocks droppable in one fall move |
| `goal_tolerance` | 1 | Manhattan distance that counts as "arrived" |
| `max_expansions` | 150_000 | Node-expansion budget before giving up |
| `time_budget_ms` | 2000 | Wall-clock ceiling for one search |
| `wall_penalty` | 2 | Extra cost per adjacent wall (0 disables) |
| `lava_policy` | `Forbidden { clearance: 2 }` | Hard lava exclusion + buffer |
| `lava_proximity_radius` / `_penalty` | 4 / 6 | Soft avoidance beyond the buffer |
| `path_seed` | 0 | Varies the route among equal-cost paths (human-like) |

## Extending

### Add a movement ability

Implement `Move` — the A\* core is untouched:

```rust
use azalea::BlockPos;
use azalea_pathfinder::local::moves::Edge;
use azalea_pathfinder::{default_moves, Move, MoveContext, MoveKind, WorldView};

pub struct SprintJumpMove;

impl Move for SprintJumpMove {
    fn candidates(&self, from: BlockPos, world: &dyn WorldView, ctx: &MoveContext, out: &mut Vec<Edge>) {
        // for each reachable target `to`:
        //   out.push(Edge { to, kind: MoveKind::Jump, cost: /* your cost */ });
    }
}

let mut moves = default_moves();
moves.push(Box::new(SprintJumpMove));
```

### Add a destination

Extend coverage by adding **data**, not code: push `Place`s (with a locraw `mode`
and/or an anchor) and `GraphEdge`s (`TravelEdge::Warp`, `TeleportPad`, or `Walk`)
into a `WorldGraph`. `route` already handles every edge type.

## Testing

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

The tests build hand-crafted grid worlds (flat floors, steps, ledges, lava strips,
sealed rooms) and assert on the *kind* of move chosen — that stairs are walked not
jumped, that a dry detour beats crossing lava, that a bot starting on lava escapes
before chasing the goal, and that unreachable goals yield a sensible partial path.

## Optional extensions

1. **Enable item teleport moves.** AOTV / Etherwarp need a line-of-sight raycast,
   range and mana checks (a new `MoveContext` field), and executor dispatch on
   `PathNode::reached_by`. The `Move` trait plumbing is already in place.
2. **Grow the place graph** with intra-island `Walk`/`TeleportPad` edges and zone
   anchors as areas get mapped.

## License

MIT © 2026 Raymond Shell
