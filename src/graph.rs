use azalea::BlockPos;

use super::types::Cost;

/// A named location the router can plan between: an island, or a zone
/// within one. Islands are recognized by the locraw `mode` field the bot
/// already parses; zones are anchored to coordinates on the same island.
#[derive(Debug, Clone)]
pub struct Place {
    /// Stable id used by `Destination::Place` and console commands.
    pub id: &'static str,
    /// locraw `mode` that identifies this place, if it is a whole island.
    pub mode: Option<&'static str>,
    /// Representative coordinates, used as a walking target for `Walk`
    /// edges and zone destinations. `None` for warp-only islands.
    pub anchor: Option<BlockPos>,
}

/// How to traverse from one place to another.
///
/// `TeleportPad`/`Walk` are extension points: the starter graph is warp-only,
/// and intra-island `Walk`/`TeleportPad` edges get added as data (with zone
/// anchors) as islands are mapped. `execute_route` already handles them, so
/// they're not yet constructed by `skyblock_default` — hence allow(dead_code).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum TravelEdge {
    /// Run `/warp <name>`. `Player::warp_to` just sends the command; the
    /// executor confirms the landing island via locraw afterwards.
    Warp { name: &'static str },
    /// Walk onto a teleport pad at this position.
    TeleportPad { pad: BlockPos },
    /// Plain walking between two places on the same island (target is the
    /// destination place's `anchor`).
    Walk,
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub edge: TravelEdge,
    pub cost: Cost,
}

/// The high-level Skyblock travel graph. Extending coverage means adding
/// data here (places, warps, pads, zone anchors) — never new code paths.
#[derive(Debug, Clone)]
pub struct WorldGraph {
    pub places: Vec<Place>,
    pub edges: Vec<GraphEdge>,
}

/// Rough cost of a warp in the same tenth-of-a-block units the local
/// planner uses: high enough that short walks win, low enough that warps
/// always beat cross-island walking (which is impossible anyway).
const WARP_COST: Cost = 200;

impl WorldGraph {
    /// Starter graph: Hub, Deep Caverns (`mining_2`), Dwarven Mines
    /// (`mining_3`), fully connected by warps. Zones within islands are
    /// added as `Walk`/`TeleportPad` edges with anchors as they get mapped.
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
