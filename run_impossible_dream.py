"""
Run CBS on the Impossible Dream map with A* as the low-level solver.
"""

from graph import TimeExpandedGraph, TENode
from planner import CBSPlanner, Drone
from parser import Parser
from engine import SimulationEngine


def compact_route(path: list[TENode]) -> list[str]:
    """Return only zone transitions from a time-expanded path."""
    zones: list[str] = []
    prev_zone = None
    for node in path:
        if node.zone != prev_zone:
            zones.append(node.zone)
            prev_zone = node.zone
    return zones


def main():
    filepath = "01_the_impossible_dream.txt"
    print(f"Parsing {filepath} ...")
    graph, coords, nb_drones, start_hub, end_hub = Parser.parse_map(filepath)

    print(f"  Zones       : {len(graph.zones)}")
    print(f"  Connections : {len(graph.connections) // 2}")
    print(f"  Drones      : {nb_drones}")
    print(f"  Start hub   : {start_hub}")
    print(f"  End hub     : {end_hub}")
    nb_drones_actual = nb_drones
    print()

    # No heuristic: use a conservative static fallback horizon.
    shortest_static = 1

    # Time horizon: generous upper bound
    max_time = shortest_static * nb_drones + 50 if shortest_static else 300
    print(f"  Time horizon (max_time): {max_time}")
    teg = TimeExpandedGraph(graph, max_time)

    # Create drones
    drones = [
        Drone(drone_id=i, start=start_hub, end=end_hub)
        for i in range(nb_drones_actual)
    ]

    # Set up CBS planner with our A* low-level solver
    planner = CBSPlanner(teg, drones)

    print(f"\nRunning CBS with {nb_drones_actual} drones ...")
    solution = planner.solve()

    if solution is None:
        print("\n  CBS returned no solution.")
        print("  (The map may be unsolvable under these constraints.)")
    else:
        total_cost = sum(path[-1].time for path in solution.values())
        makespan = max(path[-1].time for path in solution.values())
        print("\n  === RESULTS ===")
        print(f"  Total turns (objective / makespan): {makespan} turns")
        print(f"  Sum-of-Costs (secondary): {total_cost} turns")
        print()
        for drone_id in sorted(solution):
            path = solution[drone_id]
            arrival = path[-1].time
            zones = compact_route(path)
            print(f"  Drone {drone_id:2d}: arrives turn {arrival:3d}  "
                  f"  route: {' → '.join(zones)}")

        engine = SimulationEngine(
            graph=graph,
            coords=coords,
            paths=solution,
        )
        engine.run()


if __name__ == "__main__":
    main()
