# Azalea Pathfinder

Reusable block pathfinding and navigation infrastructure for Azalea clients.

## Extraction status

Phase one has been ported from HuntingMacro:

- A* and best-effort incremental planning
- walking, jumping, falling, and teleport extension points
- wall and lava costs
- dense, lock-free world snapshots
- high-level place graph and Dijkstra router
- plugin-facing goal, request, status, and settings types
- planner and routing regression tests

The live movement executor remains in HuntingMacro temporarily because it is
still coupled to combat state, target publication, watchdogs, and farm reset
policy. The next phase replaces those reads with `NavigationRequest`, pause,
status, and cancellation inputs, then moves the executor into this plugin.

## Intended ownership boundary

The hunting bot decides *where* to go and when navigation should pause or be
cancelled. This crate decides *how* to reach the supplied goal.
