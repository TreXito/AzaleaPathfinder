//! Measures planner throughput on a synthetic map the size of a real leg.
//!
//! Route quality is checked by the unit tests; this answers the other question,
//! which is how much of a route the search can afford inside one time budget.
//! Run with `cargo +nightly run --release --example planner_bench`.

use std::collections::HashMap;
use std::time::Instant;

use azalea::BlockPos;
use azalea_pathfinder::{BlockKind, MoveContext, WorldView, default_moves, find_path_best_effort};

/// A hilly open map with scattered walls, generated from a hash so it is the
/// same map on every run.
struct Terrain {
    span: i32,
    lava: bool,
}

impl Terrain {
    fn height(&self, x: i32, z: i32) -> i32 {
        let h = (x.wrapping_mul(374_761_393) ^ z.wrapping_mul(668_265_263)) as u32;
        let h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        60 + ((h >> 17) % 5) as i32
    }

    fn wall(&self, x: i32, z: i32) -> bool {
        // Long north-south walls with gaps, so the search has to route around
        // rather than walk a straight line to the goal.
        x.rem_euclid(19) == 0 && z.rem_euclid(23) != 0
    }
}

impl WorldView for Terrain {
    fn block(&self, pos: BlockPos) -> BlockKind {
        if pos.x.abs() > self.span || pos.z.abs() > self.span {
            return BlockKind::Unloaded;
        }
        let ground = self.height(pos.x, pos.z);
        if pos.y <= ground {
            // One lava pit, far off the direct line, so `Forbidden` stays armed
            // without the route ever touching it.
            if self.lava && pos.x > 40 && pos.x < 46 && pos.z > 40 && pos.z < 46 && pos.y == ground
            {
                return BlockKind::Lava;
            }
            return BlockKind::Solid;
        }
        if self.wall(pos.x, pos.z) && pos.y <= ground + 3 {
            return BlockKind::Solid;
        }
        BlockKind::Air
    }

    // Real planning always runs against a `WorldSnapshot`, which knows this
    // for certain because it just copied every block. Defaulting to `true`
    // here would measure a world the planner never sees.
    fn may_contain_lava(&self) -> bool {
        self.lava
    }
}

fn main() {
    let moves = default_moves();
    let ctx = MoveContext {
        // The point is to measure a full search, not a lucky early exit.
        max_expansions: 400_000,
        time_budget_ms: 60_000,
        ..MoveContext::default()
    };

    for lava in [false, true] {
        let world = Terrain { span: 200, lava };
        let start = BlockPos::new(-120, world.height(-120, -120) + 1, -120);
        let goal = BlockPos::new(120, world.height(120, 120) + 1, 120);

        // Warm the CPU cache with one throwaway run so the two numbers are
        // comparable however they are ordered.
        let _ = find_path_best_effort(&world, start, goal, &moves, &ctx);

        let mut samples = Vec::new();
        let mut nodes = 0;
        for _ in 0..3 {
            let began = Instant::now();
            let (path, reached) = find_path_best_effort(&world, start, goal, &moves, &ctx);
            samples.push(began.elapsed());
            nodes = path.nodes.len();
            assert!(reached, "bench map should be solvable");
        }
        samples.sort();
        let median = samples[samples.len() / 2];
        println!(
            "lava_in_world={lava:<5} median {:>7.1}ms  ({nodes} nodes)",
            median.as_secs_f64() * 1000.0
        );
    }

    // One real leg. Production never searches further than the snapshot it
    // captured, which spans 128 blocks of X and Z, so this is corner to corner
    // of the largest area the planner is ever handed.
    let world = Terrain { span: 64, lava: false };
    let start = BlockPos::new(-63, world.height(-63, -63) + 1, -63);
    let goal = BlockPos::new(63, world.height(63, 63) + 1, 63);
    let default_ctx = MoveContext::default();
    let began = Instant::now();
    let (path, reached) = find_path_best_effort(&world, start, goal, &moves, &default_ctx);
    let reach: i32 = path
        .nodes
        .last()
        .map(|n| (n.pos.x - start.x).abs() + (n.pos.z - start.z).abs())
        .unwrap_or(0);
    println!(
        "default budget: {:>7.1}ms  reached={reached}  {} nodes, {reach} blocks of progress",
        began.elapsed().as_secs_f64() * 1000.0,
        path.nodes.len(),
    );

    // Where the per-node time actually goes. Guessing wrong here is easy: the
    // lava rules look like the expensive ones and are not.
    let world = Terrain { span: 200, lava: false };
    let mut sites = Vec::new();
    for x in -60..60 {
        for z in -60..60 {
            sites.push(BlockPos::new(x, world.height(x, z) + 1, z));
        }
    }
    let ctx = MoveContext::default();
    let mut out = Vec::new();
    for (name, rule) in named_moves() {
        let began = Instant::now();
        for site in &sites {
            out.clear();
            rule.candidates(*site, &world, &ctx, &mut out);
        }
        let each = began.elapsed().as_secs_f64() * 1e9 / sites.len() as f64;
        println!("  {name:<8} {each:>8.0} ns/node");
    }
    let began = Instant::now();
    for site in &sites {
        std::hint::black_box(azalea_pathfinder::wall_proximity_penalty(*site, &world, 3));
    }
    println!(
        "  {:<8} {:>8.0} ns/node",
        "wall",
        began.elapsed().as_secs_f64() * 1e9 / sites.len() as f64
    );

    let _ = HashMap::<u8, u8>::new();
}

fn named_moves() -> Vec<(&'static str, Box<dyn azalea_pathfinder::Move>)> {
    use azalea_pathfinder::{ClimbMove, FallMove, JumpMove, ParkourMove, SwimMove, WalkMove};
    vec![
        ("walk", Box::new(WalkMove)),
        ("jump", Box::new(JumpMove)),
        ("climb", Box::new(ClimbMove)),
        ("parkour", Box::new(ParkourMove)),
        ("swim", Box::new(SwimMove)),
        ("fall", Box::new(FallMove)),
    ]
}
