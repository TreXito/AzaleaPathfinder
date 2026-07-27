use std::collections::HashSet;
use std::hint::black_box;

use azalea::BlockPos;
use azalea_pathfinder::{
    BlockGoal, BlockKind, MoveContext, SearchSession, SearchSlice, SearchStatus, WorldView,
    default_moves, find_path,
};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

struct BenchWorld {
    obstacles: HashSet<(i32, i32, i32)>,
    lava: HashSet<(i32, i32, i32)>,
}

impl BenchWorld {
    fn open() -> Self {
        Self {
            obstacles: HashSet::new(),
            lava: HashSet::new(),
        }
    }

    fn detour() -> Self {
        let mut world = Self::open();
        // A two-high wall forces a meaningful detour through either gap.
        for z in -20..=20 {
            for y in 64..=65 {
                world.obstacles.insert((48, y, z));
            }
        }
        // A nearby lava patch exercises cached hazard clearance.
        for x in 24..=28 {
            for z in 3..=7 {
                world.lava.insert((x, 64, z));
            }
        }
        world
    }
}

impl WorldView for BenchWorld {
    fn block(&self, position: BlockPos) -> BlockKind {
        let key = (position.x, position.y, position.z);
        if self.obstacles.contains(&key) || position.y == 63 {
            BlockKind::Solid
        } else if self.lava.contains(&key) {
            BlockKind::Lava
        } else {
            BlockKind::Air
        }
    }
}

fn context() -> MoveContext {
    MoveContext {
        goal_tolerance: 0,
        max_expansions: 500_000,
        time_budget_ms: 10_000,
        ..MoveContext::default()
    }
}

fn pathfinding_benchmarks(criterion: &mut Criterion) {
    let moves = default_moves();
    let context = context();
    let start = BlockPos::new(0, 64, 0);
    let goal = BlockPos::new(96, 64, 0);
    let open = BenchWorld::open();
    let detour = BenchWorld::detour();

    criterion.bench_function("astar/open_96_blocks", |bencher| {
        bencher.iter(|| {
            black_box(
                find_path(&open, start, goal, &moves, &context)
                    .expect("open benchmark route must exist"),
            )
        });
    });

    criterion.bench_function("astar/detour_hazard_96_blocks", |bencher| {
        bencher.iter(|| {
            black_box(
                find_path(&detour, start, goal, &moves, &context)
                    .expect("detour benchmark route must exist"),
            )
        });
    });

    let block_goal = BlockGoal::new(goal, 0);
    criterion.bench_function("astar/resumable_256_expansion_slices", |bencher| {
        bencher.iter_batched(
            || SearchSession::new(&detour, start, &block_goal, &moves, &context),
            |mut session| loop {
                match session.advance(SearchSlice::new(256)) {
                    SearchStatus::InProgress => {}
                    SearchStatus::Found(path) => break black_box(path),
                    status => panic!("benchmark route ended unexpectedly: {status:?}"),
                }
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, pathfinding_benchmarks);
criterion_main!(benches);
