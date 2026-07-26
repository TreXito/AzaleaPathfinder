use std::cmp::Reverse;
use std::collections::BinaryHeap;

use azalea::BlockPos;

use super::graph::{TravelEdge, WorldGraph};
use super::types::Cost;

/// One leg of a high-level route.
#[derive(Debug, Clone)]
pub struct TravelStep {
    pub to_id: &'static str,
    /// Expected locraw mode after this step, if any.
    pub to_mode: Option<&'static str>,
    /// Destination anchor for a walking step.
    pub to_anchor: Option<BlockPos>,
    pub edge: TravelEdge,
}

/// Finds the cheapest route between two places.
///
/// Returns an empty route when `from == to`, or `None` when either place is
/// invalid or unreachable.
pub fn route(graph: &WorldGraph, from: usize, to: usize) -> Option<Vec<TravelStep>> {
    let n = graph.places.len();
    if from >= n || to >= n {
        return None;
    }
    if from == to {
        return Some(Vec::new());
    }

    let mut outgoing = vec![Vec::new(); n];
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        // Caller-built graphs may contain invalid edges.
        if edge.from < n && edge.to < n {
            outgoing[edge.from].push(edge_idx);
        }
    }

    let mut dist: Vec<Option<Cost>> = vec![None; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    let mut open = BinaryHeap::new();
    dist[from] = Some(0);
    open.push(Reverse((0, from)));

    while let Some(Reverse((current_dist, current))) = open.pop() {
        if dist[current] != Some(current_dist) {
            continue;
        }
        if current == to {
            break;
        }

        for &edge_idx in &outgoing[current] {
            let e = &graph.edges[edge_idx];
            let candidate = current_dist.saturating_add(e.cost);
            if dist[e.to].is_none_or(|d| candidate < d) {
                dist[e.to] = Some(candidate);
                prev[e.to] = Some(edge_idx);
                open.push(Reverse((candidate, e.to)));
            }
        }
    }

    dist[to]?;
    let mut steps = Vec::new();
    let mut at = to;
    while at != from {
        let edge_idx = prev[at]?;
        let e = &graph.edges[edge_idx];
        let place = &graph.places[e.to];
        steps.push(TravelStep {
            to_id: place.id,
            to_mode: place.mode,
            to_anchor: place.anchor,
            edge: e.edge.clone(),
        });
        at = e.from;
    }
    steps.reverse();
    Some(steps)
}

#[cfg(test)]
mod tests {
    use crate::graph::{GraphEdge, Place};

    use super::*;

    #[test]
    fn warps_between_islands() {
        let graph = WorldGraph::skyblock_default();
        let hub = graph.place_index("hub").unwrap();
        let deep = graph.place_index("deep_caverns").unwrap();
        let dwarven = graph.place_index("dwarven_mines").unwrap();

        let steps = route(&graph, hub, deep).unwrap();
        assert_eq!(steps.len(), 1);
        assert!(matches!(steps[0].edge, TravelEdge::Warp { name: "deep" }));
        assert_eq!(steps[0].to_mode, Some("mining_2"));

        let steps = route(&graph, dwarven, deep).unwrap();
        assert_eq!(steps.len(), 1);
        assert!(matches!(steps[0].edge, TravelEdge::Warp { name: "deep" }));
    }

    #[test]
    fn chooses_the_cheapest_route_and_ignores_invalid_edges() {
        let places = vec![
            Place {
                id: "a",
                mode: None,
                anchor: None,
            },
            Place {
                id: "b",
                mode: None,
                anchor: None,
            },
            Place {
                id: "c",
                mode: None,
                anchor: None,
            },
        ];
        let walk = |from, to, cost| GraphEdge {
            from,
            to,
            edge: TravelEdge::Walk,
            cost,
        };
        let graph = WorldGraph {
            places,
            edges: vec![
                walk(0, 2, 50),
                walk(0, 1, 10),
                walk(1, 2, 10),
                walk(99, 2, 1),
                walk(0, 99, 1),
            ],
        };

        let steps = route(&graph, 0, 2).unwrap();
        assert_eq!(
            steps.iter().map(|step| step.to_id).collect::<Vec<_>>(),
            ["b", "c"]
        );
    }
}
