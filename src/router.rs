use azalea::BlockPos;

use super::graph::{TravelEdge, WorldGraph};
use super::types::Cost;

/// One leg of a high-level route, ready for the executor.
#[derive(Debug, Clone)]
pub struct TravelStep {
    pub to_id: &'static str,
    /// locraw mode expected after this step, used to verify warps landed.
    pub to_mode: Option<&'static str>,
    /// Walking target for `Walk` edges (the destination place's anchor).
    pub to_anchor: Option<BlockPos>,
    pub edge: TravelEdge,
}

/// Dijkstra over the place graph. Returns the cheapest sequence of travel
/// steps from `from` to `to` (empty if already there), or `None` if the
/// graph has no connection. The graph is tiny, so a linear-scan Dijkstra
/// is plenty.
pub fn route(graph: &WorldGraph, from: usize, to: usize) -> Option<Vec<TravelStep>> {
    let n = graph.places.len();
    if from >= n || to >= n {
        return None;
    }
    if from == to {
        return Some(Vec::new());
    }

    let mut dist: Vec<Option<Cost>> = vec![None; n];
    let mut prev: Vec<Option<usize>> = vec![None; n]; // edge index taken to reach node
    let mut done = vec![false; n];
    dist[from] = Some(0);

    loop {
        // pick the cheapest unfinished node
        let Some(current) = (0..n)
            .filter(|&i| !done[i] && dist[i].is_some())
            .min_by_key(|&i| dist[i].unwrap())
        else {
            return None; // goal unreachable
        };
        if current == to {
            break;
        }
        done[current] = true;

        for (edge_idx, e) in graph.edges.iter().enumerate() {
            // WorldGraph is public data and callers may construct it manually.
            // Ignore malformed endpoints rather than indexing past `dist`.
            if e.from != current || e.to >= n {
                continue;
            }
            let candidate = dist[current].unwrap().saturating_add(e.cost);
            if dist[e.to].is_none_or(|d| candidate < d) {
                dist[e.to] = Some(candidate);
                prev[e.to] = Some(edge_idx);
            }
        }
    }

    // walk the edge chain backwards from the goal
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
    fn already_there_is_empty_route() {
        let graph = WorldGraph::skyblock_default();
        let hub = graph.place_index("hub").unwrap();
        assert!(route(&graph, hub, hub).unwrap().is_empty());
    }

    #[test]
    fn malformed_public_edges_are_ignored_instead_of_panicking() {
        let graph = WorldGraph {
            places: vec![
                crate::Place {
                    id: "from",
                    mode: None,
                    anchor: None,
                },
                crate::Place {
                    id: "to",
                    mode: None,
                    anchor: None,
                },
            ],
            edges: vec![crate::GraphEdge {
                from: 0,
                to: usize::MAX,
                edge: TravelEdge::Walk,
                cost: 1,
            }],
        };

        assert!(route(&graph, 0, 1).is_none());
    }
}
