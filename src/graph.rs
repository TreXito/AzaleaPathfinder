use azalea::BlockPos;

use super::types::Cost;

/// A named island or zone in the travel graph.
#[derive(Debug, Clone)]
pub struct Place {
    /// Stable identifier used to look up this place.
    pub id: &'static str,
    /// Locraw mode, or `None` for a zone.
    pub mode: Option<&'static str>,
    /// Walking target, or `None` for a warp-only place.
    pub anchor: Option<BlockPos>,
}

/// How to travel between two places.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum TravelEdge {
    /// Run `/warp <name>`.
    Warp { name: &'static str },
    /// Walk onto a teleport pad at this position.
    TeleportPad { pad: BlockPos },
    /// Walk to the destination place's anchor.
    Walk,
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub edge: TravelEdge,
    pub cost: Cost,
}

/// A graph of named places and travel links.
#[derive(Debug, Clone)]
pub struct WorldGraph {
    pub places: Vec<Place>,
    pub edges: Vec<GraphEdge>,
}

/// Priced so short walks beat warps.
const WARP_COST: Cost = 200;

impl WorldGraph {
    /// Hub, Deep Caverns, and Dwarven Mines, connected by warps.
    pub fn skyblock_default() -> Self {
        let places = vec![
            Place {
                id: "hub",
                mode: Some("hub"),
                anchor: None,
            },
            Place {
                id: "deep_caverns",
                mode: Some("mining_2"),
                anchor: None,
            },
            Place {
                id: "dwarven_mines",
                mode: Some("mining_3"),
                anchor: None,
            },
        ];

        let hub = 0;
        let deep = 1;
        let dwarven = 2;

        let warp = |from: usize, to: usize, name: &'static str| GraphEdge {
            from,
            to,
            edge: TravelEdge::Warp { name },
            cost: WARP_COST,
        };

        let edges = vec![
            warp(hub, deep, "deep"),
            warp(hub, dwarven, "mines"),
            warp(deep, hub, "hub"),
            warp(deep, dwarven, "mines"),
            warp(dwarven, hub, "hub"),
            warp(dwarven, deep, "deep"),
        ];

        Self { places, edges }
    }

    pub fn place_index(&self, id: &str) -> Option<usize> {
        self.places.iter().position(|p| p.id == id)
    }

    pub fn place_by_mode(&self, mode: &str) -> Option<usize> {
        self.places.iter().position(|p| p.mode == Some(mode))
    }
}
